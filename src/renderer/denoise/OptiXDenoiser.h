/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_denoiser_tiling.h>

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "OptiXDenoiserUtil.h"


struct OptiXDenoiser
{
    struct InputData
    {
        OptixImage2D    color;
        OptixImage2D    albedo;
        OptixImage2D    normal;
        OptixImage2D    flow;
        OptixImage2D    flowtrust;
        std::vector< OptixImage2D > aovs;
    };

    struct OutputData
    {
        OptixImage2D    color;
        std::vector< OptixImage2D > aovs;
    };

    OptiXDenoiser( OptixLogCallback      logger,
                   void*                 cbdata )
                   : m_logger( logger ), m_cbdata( cbdata ) {}
    ~OptiXDenoiser() {}

    bool init( OptixDeviceContext&       context,
               CUstream                  stream,
               unsigned int              width,
               unsigned int              height,
               unsigned int              tileWidth,
               unsigned int              tileHeight,
               bool                      upscale2x,
               bool                      guideAlbedo,
               bool                      guideNormal,
               bool                      temporal,
               bool                      denoiseAlpha );

    bool denoise( const OutputData&      outData,
                  const InputData&       inData,
                  CUstream               stream,
                  float flowMulX         = 1.f,
                  float flowMulY         = 1.f,
                  bool newSequence       = true );

    bool exit();

    const OptixImage2D getInternalGuideLayer() const { return m_guideLayer.outputInternalGuideLayer; }

private:
    OptixDenoiserLayer makeLayer( const OptixImage2D& input, const OptixImage2D& output, const OptixImage2D& prevOutput );

    OptixDenoiser         m_denoiser     = nullptr;
    OptixDenoiserParams   m_params       = {};

    bool                  m_temporalMode;

    CUdeviceptr           m_scratch      = 0;
    uint32_t              m_scratch_size = 0;
    CUdeviceptr           m_state        = 0;
    uint32_t              m_state_size   = 0;

    unsigned int          m_tileWidth    = 0;
    unsigned int          m_tileHeight   = 0;
    unsigned int          m_overlap      = 0;

    OptixDenoiserGuideLayer m_guideLayer = {};
    OptixImage2D          m_previousOutput = {};
    std::vector< OptixImage2D > m_previousAOVOutput;

    OptixLogCallback      m_logger       = 0;
    void*                 m_cbdata       = 0;
};

bool OptiXDenoiser::init( OptixDeviceContext& context,
                          CUstream            stream,
                          unsigned int        width,
                          unsigned int        height,
                          unsigned int        tileWidth,
                          unsigned int        tileHeight,
                          bool                upscale2x,
                          bool                guideAlbedo,
                          bool                guideNormal,
                          bool                temporal,
                          bool                denoiseAlpha )
{
    m_temporalMode = temporal;

    m_tileWidth  = tileWidth  > 0 ? tileWidth  : width;
    m_tileHeight = tileHeight > 0 ? tileHeight : height;

    //
    // Create denoiser
    //
    OptixDenoiserOptions options = {};
    options.guideAlbedo  = guideAlbedo;
    options.guideNormal  = guideNormal;
    options.denoiseAlpha = denoiseAlpha ? OPTIX_DENOISER_ALPHA_MODE_DENOISE : OPTIX_DENOISER_ALPHA_MODE_COPY;

    OptixDenoiserModelKind modelKind;
    if( upscale2x )
        modelKind = temporal ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
    else
        modelKind = temporal ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV : OPTIX_DENOISER_MODEL_KIND_AOV;

    if( optixDenoiserCreate( context, modelKind, &options, &m_denoiser ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to create denoiser", m_cbdata );
        return false;
    }

    //
    // Allocate device memory for denoiser
    //
    OptixDenoiserSizes denoiser_sizes;

    if( optixDenoiserComputeMemoryResources(
                m_denoiser,
                m_tileWidth,
                m_tileHeight,
                &denoiser_sizes
                ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to obtain memory resources", m_cbdata );
        return false;
    }

    if( tileWidth == 0 )
    {
        m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withoutOverlapScratchSizeInBytes );
        m_overlap = 0;
    }
    else
    {
        m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withOverlapScratchSizeInBytes );
        m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
    }

    if( cudaMalloc( reinterpret_cast<void**>( &m_scratch ), m_scratch_size ) != cudaSuccess )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to allocate scratch size", m_cbdata );
        return false;
    }

    m_state_size = static_cast<uint32_t>( denoiser_sizes.stateSizeInBytes );

    if( cudaMalloc( reinterpret_cast<void**>( &m_state ), denoiser_sizes.stateSizeInBytes) != cudaSuccess )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to allocate state memory", m_cbdata );
        return false;
    }

    if( m_temporalMode )
    {
        unsigned int outScale = upscale2x ? 2 : 1;

        // Internal guide layer memory set to zero for first frame.
        void* internalMemIn  = 0;
        void* internalMemOut = 0;
        size_t internalSize = outScale * width * outScale * height * denoiser_sizes.internalGuideLayerPixelSizeInBytes;
        if( cudaMalloc( &internalMemIn, internalSize ) != cudaSuccess )
            return false;
        if( cudaMalloc( &internalMemOut, internalSize ) != cudaSuccess )
        {
            if( m_logger )
                m_logger( 2, "DENOISER", "Failed to allocate memory for internal guide layer", m_cbdata );
            return false;
        }
        if( cudaMemsetAsync( internalMemIn, 0, internalSize, stream ) != cudaSuccess )
        {
            if( m_logger )
                m_logger( 2, "DENOISER", "Failed to allocate memory for internal guide layer", m_cbdata );
            return false;
        }

        m_guideLayer.previousOutputInternalGuideLayer.data   = (CUdeviceptr)internalMemIn;
        m_guideLayer.previousOutputInternalGuideLayer.width  = outScale * width;
        m_guideLayer.previousOutputInternalGuideLayer.height = outScale * height;
        m_guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned( denoiser_sizes.internalGuideLayerPixelSizeInBytes );
        m_guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes =
			m_guideLayer.previousOutputInternalGuideLayer.width *
			m_guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes;
        m_guideLayer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

        m_guideLayer.outputInternalGuideLayer = m_guideLayer.previousOutputInternalGuideLayer;
        m_guideLayer.outputInternalGuideLayer.data = (CUdeviceptr)internalMemOut;
    }

    //
    // Setup denoiser
    //
    if( optixDenoiserSetup(
                m_denoiser,
                stream,
                m_tileWidth + 2 * m_overlap,
                m_tileHeight + 2 * m_overlap,
                m_state,
                m_state_size,
                m_scratch,
                m_scratch_size
                ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failure from call to optixDenoiserSetup", m_cbdata );
        return false;
    }

    return true;
}

inline OptixDenoiserLayer OptiXDenoiser::makeLayer( const OptixImage2D& input, const OptixImage2D& output, const OptixImage2D& prevOutput )
{
    OptixDenoiserLayer layer = {};
    layer.input = input;
    layer.output = output;
    layer.previousOutput = prevOutput;
    return layer;
}

bool OptiXDenoiser::denoise( const OutputData& outData,
                             const InputData& inData,
                             CUstream         stream,
                             float            flowMulX,
                             float            flowMulY,
                             bool             newSequence )
{
    bool upscale2xMode = bool( outData.color.width > inData.color.width );

    if( m_temporalMode )
    {
        unsigned int outScale = upscale2xMode ? 2 : 1;

        if( !compareOptixImage2D( outData.color, m_previousOutput ) )
        {
            freeOptixImage2D( m_previousOutput );
            m_previousOutput = createOptixImage2D( outScale * inData.color.width, outScale * inData.color.height, inData.color.format );
        }

        if( m_previousAOVOutput.size() != inData.aovs.size() )
        {
            for( size_t i=0; i < m_previousAOVOutput.size(); i++ )
                freeOptixImage2D( m_previousAOVOutput[i] );

            m_previousAOVOutput.resize( inData.aovs.size() );
            for( size_t i=0; i < m_previousAOVOutput.size(); i++ )
                m_previousAOVOutput[i] = {};
        }

        for( size_t i=0; i < inData.aovs.size(); i++ )
        {
            if( !compareOptixImage2D( outData.aovs[i], m_previousAOVOutput[i] ) )
            {
                freeOptixImage2D( m_previousAOVOutput[i] );
                m_previousAOVOutput[i] = createOptixImage2D( outScale * inData.aovs[i].width, outScale * inData.aovs[i].height, OPTIX_PIXEL_FORMAT_FLOAT4 );
            }
        }
    }

    if( m_temporalMode && newSequence && !upscale2xMode )
    {
        // first frame in a sequence: copy noisy image into previous denoised image (approximation)
        if( cudaMemcpyAsync( (void*)m_previousOutput.data, (void*)inData.color.data,
                             inData.color.rowStrideInBytes * inData.color.height, cudaMemcpyDeviceToDevice, stream ) != cudaSuccess )
        {
            if( m_logger )
                m_logger( 2, "DENOISER", "Failed to copy first frame", m_cbdata );
            return false;
        }
    }

    std::vector< OptixDenoiserLayer > layers;
    layers.push_back( makeLayer( inData.color, outData.color, m_previousOutput ) );

    for( size_t i=0; i < inData.aovs.size(); i++ )
    {
        if( m_temporalMode && newSequence && !upscale2xMode )
        {
            // first frame in a sequence: copy noisy image into previous denoised image (approximation)
            if( cudaMemcpyAsync( (void*)m_previousAOVOutput[i].data, (void*)inData.aovs[i].data,
                                 inData.aovs[i].rowStrideInBytes * inData.aovs[i].height, cudaMemcpyDeviceToDevice, stream ) != cudaSuccess )
            {
                if( m_logger )
                    m_logger( 2, "DENOISER", "Failed to copy aov", m_cbdata );
                return false;
            }
        }
        layers.push_back( makeLayer( inData.aovs[i], outData.aovs[i], m_previousAOVOutput[i] ) );
    }

    OptixDenoiserGuideLayer guideLayer = m_guideLayer;
    guideLayer.albedo = inData.albedo;
    guideLayer.normal = inData.normal;
    guideLayer.flow   = inData.flow;

    OptixDenoiserParams params = {};
    params.temporalModeUsePreviousLayers = !newSequence;
    params.flowMulX = flowMulX;
    params.flowMulY = flowMulY;

    if( optixUtilDenoiserInvokeTiled(
                m_denoiser,
                stream,
                &params,
                m_state,
                m_state_size,
                &guideLayer,
                layers.data(),
                static_cast<unsigned int>( layers.size() ),
                m_scratch,
                m_scratch_size,
                m_overlap,
                m_tileWidth,
                m_tileHeight
                ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failure from call to optixUtilDenoiserInvokeTiled", m_cbdata );
        return false;
    }

    // swap internal guide layers, initialize previous denoised image
    if( m_temporalMode )
    {
        if( cudaMemcpyAsync( (void*)m_previousOutput.data, (void*)outData.color.data,
                             outData.color.rowStrideInBytes * outData.color.height, cudaMemcpyDeviceToDevice, stream ) != cudaSuccess )
        {
            if( m_logger )
                m_logger( 2, "DENOISER", "Failed to copy denoised image to previous denoised image", m_cbdata );
            return false;
        }

        OptixImage2D temp = m_guideLayer.previousOutputInternalGuideLayer;
        m_guideLayer.previousOutputInternalGuideLayer = m_guideLayer.outputInternalGuideLayer;
        m_guideLayer.outputInternalGuideLayer = temp;

        for( size_t i=0; i < outData.aovs.size(); i++ )
        {
            if( cudaMemcpyAsync( (void*)m_previousAOVOutput[i].data, (void*)outData.aovs[i].data,
                                 outData.aovs[i].rowStrideInBytes * outData.aovs[i].height, cudaMemcpyDeviceToDevice, stream ) != cudaSuccess )
            {
                if( m_logger )
                    m_logger( 2, "DENOISER", "Failed to copy denoised aov to previous denoised aov", m_cbdata );
                return false;
            }
        }
    }
    return true;
}

bool OptiXDenoiser::exit()
{
    // Cleanup resources
    if( m_denoiser )
        optixDenoiserDestroy( m_denoiser );

    bool ret = true;

    if( m_scratch && cudaFree( reinterpret_cast<void*>(m_scratch) ) != cudaSuccess )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to release scratch memory", m_cbdata );
        ret = false;
    }

    if( m_state && cudaFree( reinterpret_cast<void*>(m_state) ) != cudaSuccess )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to release state memory", m_cbdata );
        ret = false;
    }

    if( !freeOptixImage2D( m_guideLayer.previousOutputInternalGuideLayer ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to release memory for previous internal guide layer", m_cbdata );
        ret = false;
    }

    if( !freeOptixImage2D( m_guideLayer.outputInternalGuideLayer ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to release memory for internal guide layer", m_cbdata );
        ret = false;
    }

    if( !freeOptixImage2D( m_previousOutput ) )
    {
        if( m_logger )
            m_logger( 2, "DENOISER", "Failed to release memory for previous output", m_cbdata );
        ret = false;
    }

    for( size_t i=0; i < m_previousAOVOutput.size(); i++ )
    {
        if( !freeOptixImage2D( m_previousAOVOutput[i] ) )
        {
            if( m_logger )
                m_logger( 2, "DENOISER", "Failed to release memory for previous aov", m_cbdata );
            ret = false;
        }
    }

    return ret;
}
