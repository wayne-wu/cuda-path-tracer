/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

static inline OptixImage2D createOptixImage2D( unsigned int width, unsigned int height, OptixPixelFormat format )
{
    OptixImage2D oi = {};

    unsigned int nChannels   = 0;
    unsigned int channelSize = 0;

    switch( format )
    {
        case OPTIX_PIXEL_FORMAT_HALF2:
            nChannels = 2;
            channelSize = sizeof( short );
            break;
        case OPTIX_PIXEL_FORMAT_HALF3:
            nChannels = 3;
            channelSize = sizeof( short );
            break;
        case OPTIX_PIXEL_FORMAT_HALF4:
            nChannels = 4;
            channelSize = sizeof( short );
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT2:
            nChannels = 2;
            channelSize = sizeof( float );
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT3:
            nChannels = 3;
            channelSize = sizeof( float );
            break;
        case OPTIX_PIXEL_FORMAT_FLOAT4:
            nChannels = 4;
            channelSize = sizeof( float );
            break;
	default:
	    break;
    }

    cudaMalloc( reinterpret_cast<void**>( &oi.data ), width * height * nChannels * channelSize );

    oi.width              = width;
    oi.height             = height;
    oi.pixelStrideInBytes = channelSize * nChannels;
    oi.rowStrideInBytes   = width * oi.pixelStrideInBytes;
    oi.format             = format;

    return oi;
}

static bool freeOptixImage2D( OptixImage2D& image )
{
    if( image.data )
        cudaFree( (void*)image.data );
    image = {};
    return true;
}

static bool compareOptixImage2D( const OptixImage2D& a, const OptixImage2D& b )
{
    return a.width == b.width && a.height == b.height && a.format == b.format;
}
