const vertexShaderWHLSL1 = `
struct Output {
    float3 vPositionW : attribute(0);
    float3 vNormalW : attribute(1);
    float2 vMainUV1 : attribute(2);
    float4 position : SV_Position;
}

struct Scene {
    float4x4 viewProjection;
    float4x4 view;
}

struct Material {
    float2 vAlbedoInfos;
    float4 vAmbientInfos;
    float2 vOpacityInfos;
    float2 vEmissiveInfos;
    float2 vLightmapInfos;
    float3 vReflectivityInfos;
    float2 vMicroSurfaceSamplerInfos;
    float2 vReflectionInfos;
    float3 vReflectionPosition;
    float3 vReflectionSize;
    float3 vBumpInfos;
    float4x4 albedoMatrix;
    float4x4 ambientMatrix;
    float4x4 opacityMatrix;
    float4x4 emissiveMatrix;
    float4x4 lightmapMatrix;
    float4x4 reflectivityMatrix;
    float4x4 microSurfaceSamplerMatrix;
    float4x4 bumpMatrix;
    float2 vTangentSpaceParams;
    float4x4 reflectionMatrix;
    float3 vReflectionColor;
    float4 vAlbedoColor;
    float4 vLightingIntensity;
    float3 vReflectionMicrosurfaceInfos;
    float pointSize;
    float4 vReflectivityColor;
    float3 vEmissiveColor;
    float4 vEyePosition;
    float3 vAmbientColor;
    float2 vDebugMode;
    float2 vClearCoatParams;
    float4 vClearCoatRefractionParams;
    float2 vClearCoatInfos;
    float4x4 clearCoatMatrix;
    float2 vClearCoatBumpInfos;
    float2 vClearCoatTangentSpaceParams;
    float4x4 clearCoatBumpMatrix;
    float4 vClearCoatTintParams;
    float clearCoatColorAtDistance;
    float2 vClearCoatTintInfos;
    float4x4 clearCoatTintMatrix;
    float3 vAnisotropy;
    float2 vAnisotropyInfos;
    float4x4 anisotropyMatrix;
    float4 vSheenColor;
    float2 vSheenInfos;
    float4x4 sheenMatrix;
    float3 vRefractionMicrosurfaceInfos;
    float4 vRefractionInfos;
    float4x4 refractionMatrix;
    float2 vThicknessInfos;
    float4x4 thicknessMatrix;
    float2 vThicknessParam;
    float3 vDiffusionDistance;
    float4 vTintColor;
    float3 vSubSurfaceIntensity;
    float3 vSphericalL00;
    float3 vSphericalL1_1;
    float3 vSphericalL10;
    float3 vSphericalL11;
    float3 vSphericalL2_2;
    float3 vSphericalL2_1;
    float3 vSphericalL20;
    float3 vSphericalL21;
    float3 vSphericalL22;
    float3 vSphericalX;
    float3 vSphericalY;
    float3 vSphericalZ;
    float3 vSphericalXX_ZZ;
    float3 vSphericalYY_ZZ;
    float3 vSphericalZZ;
    float3 vSphericalXY;
    float3 vSphericalYZ;
    float3 vSphericalZX;
}

struct Mesh {
    float4x4 world;
    float visibility;
}

float3x3 transposeMat3(float3x3 inMatrix) {
    float3 i0 = inMatrix[0];
    float3 i1 = inMatrix[1];
    float3 i2 = inMatrix[2];
    float3x3 outMatrix = float3x3(
        float3(i0.x, i1.x, i2.x),
        float3(i0.y, i1.y, i2.y),
        float3(i0.z, i1.z, i2.z)
    );
    return outMatrix;
}

float3x3 inverseMat3(float3x3 inMatrix) {
    float a00 = inMatrix[0][0], a01 = inMatrix[0][1], a02 = inMatrix[0][2];
    float a10 = inMatrix[1][0], a11 = inMatrix[1][1], a12 = inMatrix[1][2];
    float a20 = inMatrix[2][0], a21 = inMatrix[2][1], a22 = inMatrix[2][2];
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
    float det = a00 * b01 + a01 * b11 + a02 * b21;
    return float3x3(b01, -a22 * a01 + a02 * a21, a12 * a01 - a02 * a11,
        b11, a22 * a00 - a02 * a20, -a12 * a00 + a02 * a10,
        b21, -a21 * a00 + a01 * a20, a11 * a00 - a01 * a10) / det;
}

float3 toLinearSpace(float3 color) {
    return pow(color, float3(2.2, 2.2, 2.2));
}

float3 toGammaSpace(float3 color) {
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float square(float value) {
    return value * value;
}

float pow5(float value) {
    float sq = value * value;
    return sq * sq * value;
}

float getLuminance(float3 color) {
    return clamp(dot(color, float3(0.2126, 0.7152, 0.0722)), 0., 1.);
}

float getRand(float2 seed) {
    return frac(sin(dot(seed.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float dither(float2 seed, float varianceAmount) {
    float rand = getRand(seed);
    float dither = lerp(-varianceAmount / 255.0, varianceAmount / 255.0, rand);
    return dither;
}

float4 toRGBD(float3 color) {
    float maxRGB = max(max(color.x, max(color.y, color.z)), 0.0000001);
    float D = max(255.0 / maxRGB, 1.);
    D = clamp(floor(D) / 255.0, 0., 1.);
    float3 rgb = color * D;
    rgb = toGammaSpace(rgb);
    return float4(rgb, D);
}

float3 fromRGBD(float4 rgbd) {
    rgbd.xyz = toLinearSpace(rgbd.xyz);
    return rgbd.xyz / rgbd.w;
}

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1), float2 uv : attribute(2),
        constant Scene[] scene : register(b0, space0),
        Texture2D<float4> environmentBrdfSamplerTexture : register(t1, space0),
        sampler environmentBrdfSamplerSampler : register(s2, space0),
        constant Material[] material : register(b0, space1),
        constant Mesh[] mesh : register(b1, space1),
        Texture2D<float4> reflectionSamplerTexture : register(t0, space2),
        sampler reflectionSamplerSampler : register(s1, space2),
        Texture2D<float4> albedoSamplerTexture : register(t2, space2),
        sampler albedoSamplerSampler : register(s3, space2),
        Texture2D<float4> reflectivitySamplerTexture : register(t4, space2),
        sampler reflectivitySamplerSampler : register(s5, space2),
        Texture2D<float4> ambientSamplerTexture : register(t6, space2),
        sampler ambientSamplerSampler : register(s7, space2),
        Texture2D<float4> emissiveSamplerTexture : register(t8, space2),
        sampler emissiveSamplerSampler : register(s9, space2),
        Texture2D<float4> bumpSamplerTexture : register(t10, space2),
        sampler bumpSamplerSampler : register(s11, space2)) {
    Output output;
    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    float2 uvUpdated = uv;
    float4x4 finalWorld = mesh[0].world;
    output.position = mul(mul(scene[0].viewProjection, finalWorld), float4(positionUpdated, 1.0));
    float4 worldPos = mul(finalWorld, float4(positionUpdated, 1.0));
    output.vPositionW = worldPos.xyz;
    float3x3 normalWorld;
    normalWorld[0] = finalWorld[0].xyz;
    normalWorld[1] = finalWorld[1].xyz;
    normalWorld[2] = finalWorld[2].xyz;
    output.vNormalW = normalize(mul(normalWorld, normalUpdated));
    float2 uv2 = float2(0., 0.);
    output.vMainUV1 = uvUpdated;
    return output;
}
`;

const fragmentShaderWHLSL1 = `
struct Scene {
    float4x4 viewProjection;
    float4x4 view;
}

struct Material {
    float2 vAlbedoInfos;
    float4 vAmbientInfos;
    float2 vOpacityInfos;
    float2 vEmissiveInfos;
    float2 vLightmapInfos;
    float3 vReflectivityInfos;
    float2 vMicroSurfaceSamplerInfos;
    float2 vReflectionInfos;
    float3 vReflectionPosition;
    float3 vReflectionSize;
    float3 vBumpInfos;
    float4x4 albedoMatrix;
    float4x4 ambientMatrix;
    float4x4 opacityMatrix;
    float4x4 emissiveMatrix;
    float4x4 lightmapMatrix;
    float4x4 reflectivityMatrix;
    float4x4 microSurfaceSamplerMatrix;
    float4x4 bumpMatrix;
    float2 vTangentSpaceParams;
    float4x4 reflectionMatrix;
    float3 vReflectionColor;
    float4 vAlbedoColor;
    float4 vLightingIntensity;
    float3 vReflectionMicrosurfaceInfos;
    float pointSize;
    float4 vReflectivityColor;
    float3 vEmissiveColor;
    float4 vEyePosition;
    float3 vAmbientColor;
    float2 vDebugMode;
    float2 vClearCoatParams;
    float4 vClearCoatRefractionParams;
    float2 vClearCoatInfos;
    float4x4 clearCoatMatrix;
    float2 vClearCoatBumpInfos;
    float2 vClearCoatTangentSpaceParams;
    float4x4 clearCoatBumpMatrix;
    float4 vClearCoatTintParams;
    float clearCoatColorAtDistance;
    float2 vClearCoatTintInfos;
    float4x4 clearCoatTintMatrix;
    float3 vAnisotropy;
    float2 vAnisotropyInfos;
    float4x4 anisotropyMatrix;
    float4 vSheenColor;
    float2 vSheenInfos;
    float4x4 sheenMatrix;
    float3 vRefractionMicrosurfaceInfos;
    float4 vRefractionInfos;
    float4x4 refractionMatrix;
    float2 vThicknessInfos;
    float4x4 thicknessMatrix;
    float2 vThicknessParam;
    float3 vDiffusionDistance;
    float4 vTintColor;
    float3 vSubSurfaceIntensity;
    float3 vSphericalL00;
    float3 vSphericalL1_1;
    float3 vSphericalL10;
    float3 vSphericalL11;
    float3 vSphericalL2_2;
    float3 vSphericalL2_1;
    float3 vSphericalL20;
    float3 vSphericalL21;
    float3 vSphericalL22;
    float3 vSphericalX;
    float3 vSphericalY;
    float3 vSphericalZ;
    float3 vSphericalXX_ZZ;
    float3 vSphericalYY_ZZ;
    float3 vSphericalZZ;
    float3 vSphericalXY;
    float3 vSphericalYZ;
    float3 vSphericalZX;
}

struct Mesh {
    float4x4 world;
    float visibility;
}

float3x3 transposeMat3(float3x3 inMatrix) {
    float3 i0 = inMatrix[0];
    float3 i1 = inMatrix[1];
    float3 i2 = inMatrix[2];
    float3x3 outMatrix = float3x3(
        float3(i0.x, i1.x, i2.x),
        float3(i0.y, i1.y, i2.y),
        float3(i0.z, i1.z, i2.z)
    );
    return outMatrix;
}

float3x3 inverseMat3(float3x3 inMatrix) {
    float a00 = inMatrix[0][0], a01 = inMatrix[0][1], a02 = inMatrix[0][2];
    float a10 = inMatrix[1][0], a11 = inMatrix[1][1], a12 = inMatrix[1][2];
    float a20 = inMatrix[2][0], a21 = inMatrix[2][1], a22 = inMatrix[2][2];
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
    float det = a00 * b01 + a01 * b11 + a02 * b21;
    return float3x3(b01, -a22 * a01 + a02 * a21, a12 * a01 - a02 * a11,
        b11, a22 * a00 - a02 * a20, -a12 * a00 + a02 * a10,
        b21, -a21 * a00 + a01 * a20, a11 * a00 - a01 * a10) / det;
}

float3 toLinearSpace(float3 color) {
    return pow(color, float3(2.2, 2.2, 2.2));
}

float3 toGammaSpace(float3 color) {
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float square(float value) {
    return value * value;
}

float pow5(float value) {
    float sq = value * value;
    return sq * sq * value;
}

float getLuminance(float3 color) {
    return clamp(dot(color, float3(0.2126, 0.7152, 0.0722)), 0., 1.);
}

float getRand(float2 seed) {
    return frac(sin(dot(seed.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float dither(float2 seed, float varianceAmount) {
    float rand = getRand(seed);
    float dither = lerp(-varianceAmount / 255.0, varianceAmount / 255.0, rand);
    return dither;
}

float4 toRGBD(float3 color) {
    float maxRGB = max(max(color.x, max(color.y, color.z)), 0.0000001);
    float D = max(255.0 / maxRGB, 1.);
    D = clamp(floor(D) / 255.0, 0., 1.);
    float3 rgb = color * D;
    rgb = toGammaSpace(rgb);
    return float4(rgb, D);
}

float3 fromRGBD(float4 rgbd) {
    rgbd.xyz = toLinearSpace(rgbd.xyz);
    return rgbd.xyz / rgbd.w;
}

float convertRoughnessToAverageSlope(float roughness) {
    return square(roughness) + 0.0005;
}

float fresnelGrazingReflectance(float reflectance0) {
    float reflectance90 = clamp(reflectance0 * 25.0, 0.0, 1.0);
    return reflectance90;
}

float2 getAARoughnessFactors(float3 normalVector) {
    return float2(0., 0.);
}

float4 applyImageProcessing(float4 result) {
    result.xyz = toGammaSpace(result.xyz);
    result.xyz = clamp(result.xyz, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0));
    return result;
}

float3 computeEnvironmentIrradiance(float3 normal, constant Material* material) {
    return material->vSphericalL00
        + material->vSphericalL1_1 * (normal.y)
        + material->vSphericalL10 * (normal.z)
        + material->vSphericalL11 * (normal.x)
        + material->vSphericalL2_2 * (normal.y * normal.x)
        + material->vSphericalL2_1 * (normal.y * normal.z)
        + material->vSphericalL20 * ((3.0 * normal.z * normal.z) - 1.0)
        + material->vSphericalL21 * (normal.z * normal.x)
        + material->vSphericalL22 * (normal.x * normal.x - (normal.y * normal.y));
}

struct PreLightingInfo {
    float3 lightOffset;
    float lightDistanceSquared;
    float lightDistance;
    float attenuation;
    float3 L;
    float3 H;
    float NdotV;
    float NdotLUnclamped;
    float NdotL;
    float VdotH;
    float roughness;
}

PreLightingInfo computePointAndSpotPreLightingInfo(float4 lightData, float3 V, float3 N, float3 vPositionW) {
    PreLightingInfo result;
    result.lightOffset = lightData.xyz - vPositionW;
    result.lightDistanceSquared = dot(result.lightOffset, result.lightOffset);
    result.lightDistance = sqrt(result.lightDistanceSquared);
    result.L = normalize(result.lightOffset);
    result.H = normalize(V + result.L);
    result.VdotH = clamp(dot(V, result.H), 0.0, 1.0);
    result.NdotLUnclamped = dot(N, result.L);
    result.NdotL = clamp(result.NdotLUnclamped, 0.0000001, 1.0);
    return result;
}

PreLightingInfo computeDirectionalPreLightingInfo(float4 lightData, float3 V, float3 N) {
    PreLightingInfo result;
    result.lightDistance = length(-lightData.xyz);
    result.L = normalize(-lightData.xyz);
    result.H = normalize(V + result.L);
    result.VdotH = clamp(dot(V, result.H), 0.0, 1.0);
    result.NdotLUnclamped = dot(N, result.L);
    result.NdotL = clamp(result.NdotLUnclamped, 0.0000001, 1.0);
    return result;
}

PreLightingInfo computeHemisphericPreLightingInfo(float4 lightData, float3 V, float3 N) {
    PreLightingInfo result;
    result.NdotL = dot(N, lightData.xyz) * 0.5 + 0.5;
    result.NdotL = clamp(result.NdotL, 0.0000001, 1.0);
    result.NdotLUnclamped = result.NdotL;
    return result;
}

float computeDistanceLightFalloff_Standard(float3 lightOffset, float range) {
    return max(0., 1.0 - length(lightOffset) / range);
}

float computeDistanceLightFalloff_Physical(float lightDistanceSquared) {
    return 1.0 / max(lightDistanceSquared, 0.0000001);
}

float computeDistanceLightFalloff_GLTF(float lightDistanceSquared, float inverseSquaredRange) {
    float lightDistanceFalloff = 1.0 / max(lightDistanceSquared, 0.0000001);
    float factor = lightDistanceSquared * inverseSquaredRange;
    float attenuation = clamp(1.0 - factor * factor, 0.0, 1.0);
    attenuation *= attenuation;
    lightDistanceFalloff *= attenuation;
    return lightDistanceFalloff;
}

float computeDistanceLightFalloff(float3 lightOffset, float lightDistanceSquared, float range, float inverseSquaredRange) {
    return computeDistanceLightFalloff_Physical(lightDistanceSquared);
}

float computeDirectionalLightFalloff_Standard(float3 lightDirection, float3 directionToLightCenterW, float cosHalfAngle, float exponent) {
    float falloff = 0.0;
    float cosAngle = max(dot(-lightDirection, directionToLightCenterW), 0.0000001);
    if (cosAngle >= cosHalfAngle) {
        falloff = max(0., pow(cosAngle, exponent));
    }
    return falloff;
}

float computeDirectionalLightFalloff_Physical(float3 lightDirection, float3 directionToLightCenterW, float cosHalfAngle) {
    float kMinusLog2ConeAngleIntensityRatio = 6.64385618977;
    float concentrationKappa = kMinusLog2ConeAngleIntensityRatio / (1.0 - cosHalfAngle);
    float4 lightDirectionSpreadSG = float4(-lightDirection * concentrationKappa, -concentrationKappa);
    float falloff = exp2(dot(float4(directionToLightCenterW, 1.0), lightDirectionSpreadSG));
    return falloff;
}

float computeDirectionalLightFalloff_GLTF(float3 lightDirection, float3 directionToLightCenterW, float lightAngleScale, float lightAngleOffset) {
    float cd = dot(-lightDirection, directionToLightCenterW);
    float falloff = clamp(cd * lightAngleScale + lightAngleOffset, 0.0, 1.0);
    falloff *= falloff;
    return falloff;
}

float computeDirectionalLightFalloff(float3 lightDirection, float3 directionToLightCenterW, float cosHalfAngle, float exponent, float lightAngleScale, float lightAngleOffset) {
    return computeDirectionalLightFalloff_Physical(lightDirection, directionToLightCenterW, cosHalfAngle);
}

float3 getEnergyConservationFactor(float3 specularEnvironmentR0, float3 environmentBrdf) {
    return float3(1.0, 1.0, 1.0) + specularEnvironmentR0 * (1.0 / environmentBrdf.y - 1.0);
}

float3 getBRDFLookup(float NdotV, float perceptualRoughness, Texture2D<float4> environmentBrdfSamplerTexture, sampler environmentBrdfSamplerSampler) {
    float2 UV = float2(NdotV, perceptualRoughness);
    float4 brdfLookup = Sample(environmentBrdfSamplerTexture, environmentBrdfSamplerSampler, UV);
    return brdfLookup.xyz;
}

float3 getReflectanceFromBRDFLookup(float3 specularEnvironmentR0, float3 environmentBrdf) {
    float3 reflectance = lerp(environmentBrdf.xxx, environmentBrdf.yyy, specularEnvironmentR0);
    return reflectance;
}

float3 fresnelSchlickGGX(float VdotH, float3 reflectance0, float3 reflectance90) {
    return reflectance0 + (reflectance90 - reflectance0) * pow5(1.0 - VdotH);
}

float fresnelSchlickGGX(float VdotH, float reflectance0, float reflectance90) {
    return reflectance0 + (reflectance90 - reflectance0) * pow5(1.0 - VdotH);
}

float normalDistributionFunction_TrowbridgeReitzGGX(float NdotH, float alphaG) {
    float a2 = square(alphaG);
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.1415926535897932384626433832795 * d * d);
}

float smithVisibility_GGXCorrelated(float NdotL, float NdotV, float alphaG) {
    float a2 = alphaG * alphaG;
    float GGXV = NdotL * sqrt(NdotV * (NdotV - a2 * NdotV) + a2);
    float GGXL = NdotV * sqrt(NdotL * (NdotL - a2 * NdotL) + a2);
    return 0.5 / (GGXV + GGXL);
}

float diffuseBRDF_Burley(float NdotL, float NdotV, float VdotH, float roughness) {
    float diffuseFresnelNV = pow5(clamp(1.0 - NdotL, 0.0000001, 1.0));
    float diffuseFresnelNL = pow5(clamp(1.0 - NdotV, 0.0000001, 1.0));
    float diffuseFresnel90 = 0.5 + 2.0 * VdotH * VdotH * roughness;
    float fresnel = (1.0 + (diffuseFresnel90 - 1.0) * diffuseFresnelNL) *
        (1.0 + (diffuseFresnel90 - 1.0) * diffuseFresnelNV);
    return fresnel / 3.1415926535897932384626433832795;
}

struct LightingInfo {
    float3 diffuse;
}

float adjustRoughnessFromLightProperties(float roughness, float lightRadius, float lightDistance) {
    float lightRoughness = lightRadius / lightDistance;
    float totalRoughness = clamp(lightRoughness + roughness, 0.0, 1.0);
    return totalRoughness;
}

float3 computeHemisphericDiffuseLighting(PreLightingInfo info, float3 lightColor, float3 groundColor) {
    return lerp(groundColor, lightColor, float3(info.NdotL, info.NdotL, info.NdotL));
}

float3 computeDiffuseLighting(PreLightingInfo info, float3 lightColor) {
    float diffuseTerm = diffuseBRDF_Burley(info.NdotL, info.NdotV, info.VdotH, info.roughness);
    return diffuseTerm * info.attenuation * info.NdotL * lightColor;
}

float2 computeProjectionTextureDiffuseLightingUV(float4x4 textureProjectionMatrix, float3 vPositionW) {
    float4 strq = mul(textureProjectionMatrix, float4(vPositionW, 1.0));
    strq /= strq.w;
    return strq.xy;
}

float getLodFromAlphaG(float cubeMapDimensionPixels, float microsurfaceAverageSlope) {
    float microsurfaceAverageSlopeTexels = cubeMapDimensionPixels * microsurfaceAverageSlope;
    float lod = log2(microsurfaceAverageSlopeTexels);
    return lod;
}

float getLinearLodFromRoughness(float cubeMapDimensionPixels, float roughness) {
    float lod = log2(cubeMapDimensionPixels) * roughness;
    return lod;
}

float environmentRadianceOcclusion(float ambientOcclusion, float NdotVUnclamped) {
    float temp = NdotVUnclamped + ambientOcclusion;
    return clamp(square(temp) - 1.0 + ambientOcclusion, 0.0, 1.0);
}

float environmentHorizonOcclusion(float3 view, float3 normal) {
    float3 reflection = reflect(view, normal);
    float temp = clamp(1.0 + 1.1 * dot(reflection, normal), 0.0, 1.0);
    return square(temp);
}

float3 perturbNormal(float3x3 cotangentFrame, float3 textureSample, float scale) {
    textureSample = textureSample * 2.0 - 1.0;
    textureSample = normalize(textureSample * float3(scale, scale, 1.0));
    return normalize(mul(cotangentFrame, textureSample));
}

float3x3 cotangent_frame(float3 normal, float3 p, float2 uv, float2 tangentSpaceParams, bool frontFace) {
    uv = frontFace ? uv : -uv;
    float3 dp1 = ddx(p);
    float3 dp2 = ddy(p);
    float2 duv1 = ddx(uv);
    float2 duv2 = ddy(uv);
    float3 dp2perp = cross(dp2, normal);
    float3 dp1perp = cross(normal, dp1);
    float3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
    tangent *= tangentSpaceParams.x;
    bitangent *= tangentSpaceParams.y;
    float invmax = rsqrt(max(dot(tangent, tangent), dot(bitangent, bitangent)));
    return float3x3(tangent * invmax, bitangent * invmax, normal);
}

float3 perturbNormal(float3x3 cotangentFrame, float2 uv, Texture2D<float4> bumpSamplerTexture, sampler bumpSamplerSampler, float3 vBumpInfos) {
    return perturbNormal(cotangentFrame, Sample(bumpSamplerTexture, bumpSamplerSampler, uv).xyz, vBumpInfos.y);
}

float3 perturbNormal(float3x3 cotangentFrame, float3 color, float3 vBumpInfos) {
    return perturbNormal(cotangentFrame, color, vBumpInfos.y);
}

float3 parallaxCorrectNormal(float3 vertexPos, float3 origVec, float3 cubeSize, float3 cubePos) {
    float3 invOrigVec = float3(1.0, 1.0, 1.0) / origVec;
    float3 halfSize = cubeSize * 0.5;
    float3 intersecAtMaxPlane = (cubePos + halfSize - vertexPos) * invOrigVec;
    float3 intersecAtMinPlane = (cubePos - halfSize - vertexPos) * invOrigVec;
    float3 largestIntersec = max(intersecAtMaxPlane, intersecAtMinPlane);
    float distance = min(min(largestIntersec.x, largestIntersec.y), largestIntersec.z);
    float3 intersectPositionWS = vertexPos + origVec * distance;
    return intersectPositionWS - cubePos;
}

float3 computeFixedEquirectangularCoords(float4 worldPos, float3 worldNormal, float3 direction) {
    float lon = atan2(direction.z, direction.x);
    float lat = acos(direction.y);
    float2 sphereCoords = float2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return float3(s, t, 0);
}

float3 computeMirroredFixedEquirectangularCoords(float4 worldPos, float3 worldNormal, float3 direction) {
    float lon = atan2(direction.z, direction.x);
    float lat = acos(direction.y);
    float2 sphereCoords = float2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return float3(1.0 - s, t, 0);
}

float3 computeEquirectangularCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix) {
    float3 cameraToVertex = normalize(worldPos.xyz - eyePosition);
    float3 r = normalize(reflect(cameraToVertex, worldNormal));
    r = mul(reflectionMatrix, float4(r, 0)).xyz;
    float lon = atan2(r.z, r.x);
    float lat = acos(r.y);
    float2 sphereCoords = float2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return float3(s, t, 0);
}

float3 computeSphericalCoords(float4 worldPos, float3 worldNormal, float4x4 view, float4x4 reflectionMatrix) {
    float3 viewDir = normalize(mul(view, worldPos).xyz);
    float3 viewNormal = normalize(mul(view, float4(worldNormal, 0.0)).xyz);
    float3 r = reflect(viewDir, viewNormal);
    r = mul(reflectionMatrix, float4(r, 0)).xyz;
    r.z = r.z - 1.0;
    float m = 2.0 * length(r);
    return float3(r.x / m + 0.5, 1.0 - r.y / m - 0.5, 0);
}

float3 computePlanarCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix) {
    float3 viewDir = worldPos.xyz - eyePosition;
    float3 coords = normalize(reflect(viewDir, worldNormal));
    return mul(reflectionMatrix, float4(coords, 1)).xyz;
}

float3 computeCubicCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix) {
    float3 viewDir = normalize(worldPos.xyz - eyePosition);
    float3 coords = reflect(viewDir, worldNormal);
    coords = mul(reflectionMatrix, float4(coords, 0)).xyz;
    return coords;
}

float3 computeCubicLocalCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix, float3 reflectionSize, float3 reflectionPosition) {
    float3 viewDir = normalize(worldPos.xyz - eyePosition);
    float3 coords = reflect(viewDir, worldNormal);
    coords = parallaxCorrectNormal(worldPos.xyz, coords, reflectionSize, reflectionPosition);
    coords = mul(reflectionMatrix, float4(coords, 0)).xyz;
    return coords;
}

float3 computeProjectionCoords(float4 worldPos, float4x4 view, float4x4 reflectionMatrix) {
    return mul(reflectionMatrix, mul(view, worldPos)).xyz;
}

float3 computeSkyBoxCoords(float3 positionW, float4x4 reflectionMatrix) {
    return mul(reflectionMatrix, float4(positionW, 0)).xyz;
}

float3 computeReflectionCoords(float4 worldPos, float3 worldNormal, float4 vEyePosition, float4x4 reflectionMatrix) {
    return computeCubicCoords(worldPos, worldNormal, vEyePosition.xyz, reflectionMatrix);
}

fragment float4 main(float3 vPositionW : attribute(0), float3 vNormalW : attribute(1), float2 vMainUV1 : attribute(2),
        constant Scene[] scene : register(b0, space0),
        Texture2D<float4> environmentBrdfSamplerTexture : register(t1, space0),
        sampler environmentBrdfSamplerSampler : register(s2, space0),
        constant Material[] material : register(b0, space1),
        constant Mesh[] mesh : register(b1, space1),
        Texture2D<float4> reflectionSamplerTexture : register(t0, space2),
        sampler reflectionSamplerSampler : register(s1, space2),
        Texture2D<float4> albedoSamplerTexture : register(t2, space2),
        sampler albedoSamplerSampler : register(s3, space2),
        Texture2D<float4> reflectivitySamplerTexture : register(t4, space2),
        sampler reflectivitySamplerSampler : register(s5, space2),
        Texture2D<float4> ambientSamplerTexture : register(t6, space2),
        sampler ambientSamplerSampler : register(s7, space2),
        Texture2D<float4> emissiveSamplerTexture : register(t8, space2),
        sampler emissiveSamplerSampler : register(s9, space2),
        Texture2D<float4> bumpSamplerTexture : register(t10, space2),
        sampler bumpSamplerSampler : register(s11, space2),
        bool frontFace : SV_IsFrontFace) : SV_Target 0 {
    float3 viewDirectionW = normalize(material[0].vEyePosition.xyz - vPositionW);
    float3 normalW = normalize(vNormalW);
    float2 uvOffset = float2(0.0, 0.0);
    float normalScale = 1.0;
    float3x3 TBN = cotangent_frame(normalW * normalScale, vPositionW, vMainUV1, material[0].vTangentSpaceParams, frontFace);
    normalW = perturbNormal(TBN, vMainUV1 + uvOffset, bumpSamplerTexture, bumpSamplerSampler, material[0].vBumpInfos);
    float3 surfaceAlbedo = material[0].vAlbedoColor.xyz;
    float alpha = material[0].vAlbedoColor.w;
    float4 albedoTexture = Sample(albedoSamplerTexture, albedoSamplerSampler, vMainUV1 + uvOffset);
    surfaceAlbedo *= toLinearSpace(albedoTexture.xyz);
    surfaceAlbedo *= material[0].vAlbedoInfos.y;
    float3 ambientOcclusionColor = float3(1., 1., 1.);
    float3 ambientOcclusionColorMap = Sample(ambientSamplerTexture, ambientSamplerSampler, vMainUV1 + uvOffset).xyz * material[0].vAmbientInfos.y;
    ambientOcclusionColorMap = float3(ambientOcclusionColorMap.x, ambientOcclusionColorMap.x, ambientOcclusionColorMap.x);
    ambientOcclusionColor = lerp(ambientOcclusionColor, ambientOcclusionColorMap, float3(material[0].vAmbientInfos.z, material[0].vAmbientInfos.z, material[0].vAmbientInfos.z));
    float microSurface = material[0].vReflectivityColor.w;
    float3 surfaceReflectivityColor = material[0].vReflectivityColor.xyz;
    float2 metallicRoughness = surfaceReflectivityColor.xy;
    float4 surfaceMetallicColorMap = Sample(reflectivitySamplerTexture, reflectivitySamplerSampler, vMainUV1 + uvOffset);
    metallicRoughness.x *= surfaceMetallicColorMap.z;
    metallicRoughness.y *= surfaceMetallicColorMap.y;
    microSurface = 1.0 - metallicRoughness.y;
    float3 baseColor = surfaceAlbedo;
    float3 DefaultSpecularReflectanceDielectric = float3(0.04, 0.04, 0.04);
    surfaceAlbedo = lerp(baseColor * (1.0 - DefaultSpecularReflectanceDielectric.x), float3(0., 0., 0.), float3(metallicRoughness.x, metallicRoughness.x, metallicRoughness.x));
    surfaceReflectivityColor = lerp(DefaultSpecularReflectanceDielectric, baseColor, float3(metallicRoughness.x, metallicRoughness.x, metallicRoughness.x));
    microSurface = clamp(microSurface, 0.0, 1.0);
    float roughness = 1. - microSurface;
    float NdotVUnclamped = dot(normalW, viewDirectionW);
    float NdotV = abs(NdotVUnclamped) + 0.0000001;
    float alphaG = convertRoughnessToAverageSlope(roughness);
    float2 AARoughnessFactors = getAARoughnessFactors(normalW.xyz);
    alphaG += AARoughnessFactors.y;
    float4 environmentRadiance = float4(0.,0., 0., 0.);
    float3 environmentIrradiance = float3(0., 0., 0.);
    float3 reflectionVector = computeReflectionCoords(float4(vPositionW, 1.0), normalW, material[0].vEyePosition, material[0].reflectionMatrix);
    float3 reflectionCoords = reflectionVector;
    float reflectionLOD = getLodFromAlphaG(material[0].vReflectionMicrosurfaceInfos.x, alphaG);
    reflectionLOD = reflectionLOD * material[0].vReflectionMicrosurfaceInfos.y + material[0].vReflectionMicrosurfaceInfos.z;
    float requestedReflectionLOD = reflectionLOD;
    environmentRadiance = Sample(reflectionSamplerTexture, reflectionSamplerSampler, reflectionCoords.xy); //SampleLevel(reflectionSamplerTexture, reflectionSamplerSampler, reflectionCoords, requestedReflectionLOD);
    environmentRadiance.xyz = fromRGBD(environmentRadiance);
    float3 irradianceVector = mul(material[0].reflectionMatrix, float4(normalW, 0)).xyz;
    environmentIrradiance = computeEnvironmentIrradiance(irradianceVector, &material[0]);
    environmentRadiance.xyz *= material[0].vReflectionInfos.x;
    environmentRadiance.xyz *= material[0].vReflectionColor.xyz;
    environmentIrradiance *= material[0].vReflectionColor.xyz;
    float reflectance = max(max(surfaceReflectivityColor.x, surfaceReflectivityColor.y), surfaceReflectivityColor.z);
    float reflectance90 = fresnelGrazingReflectance(reflectance);
    float3 specularEnvironmentR0 = surfaceReflectivityColor.xyz;
    float3 specularEnvironmentR90 = float3(1.0, 1.0, 1.0) * reflectance90;
    float3 environmentBrdf = getBRDFLookup(NdotV, roughness, environmentBrdfSamplerTexture, environmentBrdfSamplerSampler);
    float3 energyConservationFactor = getEnergyConservationFactor(specularEnvironmentR0, environmentBrdf);
    float3 diffuseBase = float3(0., 0., 0.);
    PreLightingInfo preInfo;
    LightingInfo info;
    float shadow = 1.;
    float3 specularEnvironmentReflectance = getReflectanceFromBRDFLookup(specularEnvironmentR0, environmentBrdf);
    float ambientMonochrome = ambientOcclusionColor.x;
    float seo = environmentRadianceOcclusion(ambientMonochrome, NdotVUnclamped);
    specularEnvironmentReflectance *= seo;
    float eho = environmentHorizonOcclusion(-viewDirectionW, normalW);
    specularEnvironmentReflectance *= eho;
    float3 finalIrradiance = environmentIrradiance;
    finalIrradiance *= surfaceAlbedo.xyz;
    float3 finalRadiance = environmentRadiance.xyz;
    finalRadiance *= specularEnvironmentReflectance;
    float3 finalRadianceScaled = finalRadiance * material[0].vLightingIntensity.z;
    finalRadianceScaled *= energyConservationFactor;
    float3 finalDiffuse = diffuseBase;
    finalDiffuse *= surfaceAlbedo.xyz;
    finalDiffuse = max(finalDiffuse, float3(0.0, 0.0, 0.0));
    float3 finalAmbient = material[0].vAmbientColor;
    finalAmbient *= surfaceAlbedo.xyz;
    float3 finalEmissive = material[0].vEmissiveColor;
    float3 emissiveColorTex = Sample(emissiveSamplerTexture, emissiveSamplerSampler, vMainUV1 + uvOffset).xyz;
    finalEmissive *= toLinearSpace(emissiveColorTex);
    finalEmissive *= material[0].vEmissiveInfos.y;
    float3 ambientOcclusionForDirectDiffuse = lerp(float3(1., 1., 1.), ambientOcclusionColor, float3(material[0].vAmbientInfos.w, material[0].vAmbientInfos.w, material[0].vAmbientInfos.w));
    float4 finalColor = float4(
        finalAmbient * ambientOcclusionColor +
        finalDiffuse * ambientOcclusionForDirectDiffuse * material[0].vLightingIntensity.x +
        finalIrradiance * ambientOcclusionColor * material[0].vLightingIntensity.z +
        finalRadianceScaled +
        finalEmissive * material[0].vLightingIntensity.y,
        alpha);
    finalColor = max(finalColor, float4(0.0, 0.0, 0.0, 0.0));
    finalColor = applyImageProcessing(finalColor);
    finalColor.w *= mesh[0].visibility;
    return finalColor;
}
`;

const vertexShaderWHLSL2 = `
struct Output {
    float3 vPositionW : attribute(0);
    float3 vNormalW : attribute(1);
    float3 vPositionUVW : attribute(2);
    float4 position : SV_Position;
}

struct Scene {
    float4x4 viewProjection;
    float4x4 view;
}

struct Material {
    float2 vAlbedoInfos;
    float4 vAmbientInfos;
    float2 vOpacityInfos;
    float2 vEmissiveInfos;
    float2 vLightmapInfos;
    float3 vReflectivityInfos;
    float2 vMicroSurfaceSamplerInfos;
    float2 vReflectionInfos;
    float3 vReflectionPosition;
    float3 vReflectionSize;
    float3 vBumpInfos;
    float4x4 albedoMatrix;
    float4x4 ambientMatrix;
    float4x4 opacityMatrix;
    float4x4 emissiveMatrix;
    float4x4 lightmapMatrix;
    float4x4 reflectivityMatrix;
    float4x4 microSurfaceSamplerMatrix;
    float4x4 bumpMatrix;
    float2 vTangentSpaceParams;
    float4x4 reflectionMatrix;
    float3 vReflectionColor;
    float4 vAlbedoColor;
    float4 vLightingIntensity;
    float3 vReflectionMicrosurfaceInfos;
    float pointSize;
    float4 vReflectivityColor;
    float3 vEmissiveColor;
    float4 vEyePosition;
    float3 vAmbientColor;
    float2 vDebugMode;
    float2 vClearCoatParams;
    float4 vClearCoatRefractionParams;
    float2 vClearCoatInfos;
    float4x4 clearCoatMatrix;
    float2 vClearCoatBumpInfos;
    float2 vClearCoatTangentSpaceParams;
    float4x4 clearCoatBumpMatrix;
    float4 vClearCoatTintParams;
    float clearCoatColorAtDistance;
    float2 vClearCoatTintInfos;
    float4x4 clearCoatTintMatrix;
    float3 vAnisotropy;
    float2 vAnisotropyInfos;
    float4x4 anisotropyMatrix;
    float4 vSheenColor;
    float2 vSheenInfos;
    float4x4 sheenMatrix;
    float3 vRefractionMicrosurfaceInfos;
    float4 vRefractionInfos;
    float4x4 refractionMatrix;
    float2 vThicknessInfos;
    float4x4 thicknessMatrix;
    float2 vThicknessParam;
    float3 vDiffusionDistance;
    float4 vTintColor;
    float3 vSubSurfaceIntensity;
    float3 vSphericalL00;
    float3 vSphericalL1_1;
    float3 vSphericalL10;
    float3 vSphericalL11;
    float3 vSphericalL2_2;
    float3 vSphericalL2_1;
    float3 vSphericalL20;
    float3 vSphericalL21;
    float3 vSphericalL22;
    float3 vSphericalX;
    float3 vSphericalY;
    float3 vSphericalZ;
    float3 vSphericalXX_ZZ;
    float3 vSphericalYY_ZZ;
    float3 vSphericalZZ;
    float3 vSphericalXY;
    float3 vSphericalYZ;
    float3 vSphericalZX;
}

struct Mesh {
    float4x4 world;
    float visibility;
}

float3x3 transposeMat3(float3x3 inMatrix) {
    float3 i0 = inMatrix[0];
    float3 i1 = inMatrix[1];
    float3 i2 = inMatrix[2];
    float3x3 outMatrix = float3x3(
        float3(i0.x, i1.x, i2.x),
        float3(i0.y, i1.y, i2.y),
        float3(i0.z, i1.z, i2.z)
    );
    return outMatrix;
}

float3x3 inverseMat3(float3x3 inMatrix) {
    float a00 = inMatrix[0][0], a01 = inMatrix[0][1], a02 = inMatrix[0][2];
    float a10 = inMatrix[1][0], a11 = inMatrix[1][1], a12 = inMatrix[1][2];
    float a20 = inMatrix[2][0], a21 = inMatrix[2][1], a22 = inMatrix[2][2];
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
    float det = a00 * b01 + a01 * b11 + a02 * b21;
    return float3x3(b01, -a22 * a01 + a02 * a21, a12 * a01 - a02 * a11,
        b11, a22 * a00 - a02 * a20, -a12 * a00 + a02 * a10,
        b21, -a21 * a00 + a01 * a20, a11 * a00 - a01 * a10) / det;
}

float3 toLinearSpace(float3 color) {
    return pow(color, float3(2.2, 2.2, 2.2));
}

float3 toGammaSpace(float3 color) {
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float square(float value) {
    return value * value;
}

float pow5(float value) {
    float sq = value * value;
    return sq * sq * value;
}

float getLuminance(float3 color) {
    return clamp(dot(color, float3(0.2126, 0.7152, 0.0722)), 0., 1.);
}

float getRand(float2 seed) {
    return frac(sin(dot(seed.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float dither(float2 seed, float varianceAmount) {
    float rand = getRand(seed);
    float dither = lerp(-varianceAmount / 255.0, varianceAmount / 255.0, rand);
    return dither;
}

float4 toRGBD(float3 color) {
    float maxRGB = max(max(color.x, max(color.y, color.z)), 0.0000001);
    float D = max(255.0 / maxRGB, 1.);
    D = clamp(floor(D) / 255.0, 0., 1.);
    float3 rgb = color * D;
    rgb = toGammaSpace(rgb);
    return float4(rgb, D);
}

float3 fromRGBD(float4 rgbd) {
    rgbd.xyz = toLinearSpace(rgbd.xyz);
    return rgbd.xyz / rgbd.w;
}

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1),
        constant Scene[] scene : register(b0, space0),
        Texture2D<float4> environmentBrdfSamplerTexture : register(t1, space0),
        sampler environmentBrdfSamplerSampler : register(s2, space0),
        constant Material[] material : register(b0, space1),
        constant Mesh[] mesh : register(b1, space1),
        Texture2D<float4> reflectionSamplerTexture : register(t0, space2),
        sampler reflectionSamplerSampler : register(s1, space2)) {
    Output output;
    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    output.vPositionUVW = positionUpdated;
    float4x4 finalWorld = mesh[0].world;
    output.position = mul(mul(scene[0].viewProjection, finalWorld), float4(positionUpdated, 1.0));
    float4 worldPos = mul(finalWorld, float4(positionUpdated, 1.0));
    output.vPositionW = worldPos.xyz;
    float3x3 normalWorld;
    normalWorld[0] = finalWorld[0].xyz;
    normalWorld[1] = finalWorld[1].xyz;
    normalWorld[2] = finalWorld[2].xyz;
    output.vNormalW = normalize(mul(normalWorld, normalUpdated));
    float2 uvUpdated;
    float2 uv2;
    return output;
}
`;

const fragmentShaderWHLSL2 = `
struct Scene {
    float4x4 viewProjection;
    float4x4 view;
}

struct Material {
    float2 vAlbedoInfos;
    float4 vAmbientInfos;
    float2 vOpacityInfos;
    float2 vEmissiveInfos;
    float2 vLightmapInfos;
    float3 vReflectivityInfos;
    float2 vMicroSurfaceSamplerInfos;
    float2 vReflectionInfos;
    float3 vReflectionPosition;
    float3 vReflectionSize;
    float3 vBumpInfos;
    float4x4 albedoMatrix;
    float4x4 ambientMatrix;
    float4x4 opacityMatrix;
    float4x4 emissiveMatrix;
    float4x4 lightmapMatrix;
    float4x4 reflectivityMatrix;
    float4x4 microSurfaceSamplerMatrix;
    float4x4 bumpMatrix;
    float2 vTangentSpaceParams;
    float4x4 reflectionMatrix;
    float3 vReflectionColor;
    float4 vAlbedoColor;
    float4 vLightingIntensity;
    float3 vReflectionMicrosurfaceInfos;
    float pointSize;
    float4 vReflectivityColor;
    float3 vEmissiveColor;
    float4 vEyePosition;
    float3 vAmbientColor;
    float2 vDebugMode;
    float2 vClearCoatParams;
    float4 vClearCoatRefractionParams;
    float2 vClearCoatInfos;
    float4x4 clearCoatMatrix;
    float2 vClearCoatBumpInfos;
    float2 vClearCoatTangentSpaceParams;
    float4x4 clearCoatBumpMatrix;
    float4 vClearCoatTintParams;
    float clearCoatColorAtDistance;
    float2 vClearCoatTintInfos;
    float4x4 clearCoatTintMatrix;
    float3 vAnisotropy;
    float2 vAnisotropyInfos;
    float4x4 anisotropyMatrix;
    float4 vSheenColor;
    float2 vSheenInfos;
    float4x4 sheenMatrix;
    float3 vRefractionMicrosurfaceInfos;
    float4 vRefractionInfos;
    float4x4 refractionMatrix;
    float2 vThicknessInfos;
    float4x4 thicknessMatrix;
    float2 vThicknessParam;
    float3 vDiffusionDistance;
    float4 vTintColor;
    float3 vSubSurfaceIntensity;
    float3 vSphericalL00;
    float3 vSphericalL1_1;
    float3 vSphericalL10;
    float3 vSphericalL11;
    float3 vSphericalL2_2;
    float3 vSphericalL2_1;
    float3 vSphericalL20;
    float3 vSphericalL21;
    float3 vSphericalL22;
    float3 vSphericalX;
    float3 vSphericalY;
    float3 vSphericalZ;
    float3 vSphericalXX_ZZ;
    float3 vSphericalYY_ZZ;
    float3 vSphericalZZ;
    float3 vSphericalXY;
    float3 vSphericalYZ;
    float3 vSphericalZX;
}

struct Mesh {
    float4x4 world;
    float visibility;
}

float3x3 transposeMat3(float3x3 inMatrix) {
    float3 i0 = inMatrix[0];
    float3 i1 = inMatrix[1];
    float3 i2 = inMatrix[2];
    float3x3 outMatrix = float3x3(
        float3(i0.x, i1.x, i2.x),
        float3(i0.y, i1.y, i2.y),
        float3(i0.z, i1.z, i2.z)
    );
    return outMatrix;
}

float3x3 inverseMat3(float3x3 inMatrix) {
    float a00 = inMatrix[0][0], a01 = inMatrix[0][1], a02 = inMatrix[0][2];
    float a10 = inMatrix[1][0], a11 = inMatrix[1][1], a12 = inMatrix[1][2];
    float a20 = inMatrix[2][0], a21 = inMatrix[2][1], a22 = inMatrix[2][2];
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
    float det = a00 * b01 + a01 * b11 + a02 * b21;
    return float3x3(b01, -a22 * a01 + a02 * a21, a12 * a01 - a02 * a11,
        b11, a22 * a00 - a02 * a20, -a12 * a00 + a02 * a10,
        b21, -a21 * a00 + a01 * a20, a11 * a00 - a01 * a10) / det;
}

float3 toLinearSpace(float3 color) {
    return pow(color, float3(2.2, 2.2, 2.2));
}

float3 toGammaSpace(float3 color) {
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

float square(float value) {
    return value * value;
}

float pow5(float value) {
    float sq = value * value;
    return sq * sq * value;
}

float getLuminance(float3 color) {
    return clamp(dot(color, float3(0.2126, 0.7152, 0.0722)), 0., 1.);
}

float getRand(float2 seed) {
    return frac(sin(dot(seed.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float dither(float2 seed, float varianceAmount) {
    float rand = getRand(seed);
    float dither = lerp(-varianceAmount / 255.0, varianceAmount / 255.0, rand);
    return dither;
}

float4 toRGBD(float3 color) {
    float maxRGB = max(max(color.x, max(color.y, color.z)), 0.0000001);
    float D = max(255.0 / maxRGB, 1.);
    D = clamp(floor(D) / 255.0, 0., 1.);
    float3 rgb = color * D;
    rgb = toGammaSpace(rgb);
    return float4(rgb, D);
}

float3 fromRGBD(float4 rgbd) {
    rgbd.xyz = toLinearSpace(rgbd.xyz);
    return rgbd.xyz / rgbd.w;
}

float convertRoughnessToAverageSlope(float roughness) {
    return square(roughness) + 0.0005;
}

float fresnelGrazingReflectance(float reflectance0) {
    float reflectance90 = clamp(reflectance0 * 25.0, 0.0, 1.0);
    return reflectance90;
}

float2 getAARoughnessFactors(float3 normalVector) {
    return float2(0., 0.);
}

float4 applyImageProcessing(float4 result) {
    result.xyz = toGammaSpace(result.xyz);
    result.xyz = clamp(result.xyz, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0));
    return result;
}

struct PreLightingInfo {
    float3 lightOffset;
    float lightDistanceSquared;
    float lightDistance;
    float attenuation;
    float3 L;
    float3 H;
    float NdotV;
    float NdotLUnclamped;
    float NdotL;
    float VdotH;
    float roughness;
}

PreLightingInfo computePointAndSpotPreLightingInfo(float4 lightData, float3 V, float3 N, float3 vPositionW) {
    PreLightingInfo result;
    result.lightOffset = lightData.xyz - vPositionW;
    result.lightDistanceSquared = dot(result.lightOffset, result.lightOffset);
    result.lightDistance = sqrt(result.lightDistanceSquared);
    result.L = normalize(result.lightOffset);
    result.H = normalize(V + result.L);
    result.VdotH = clamp(dot(V, result.H), 0.0, 1.0);
    result.NdotLUnclamped = dot(N, result.L);
    result.NdotL = clamp(result.NdotLUnclamped, 0.0000001, 1.0);
    return result;
}

PreLightingInfo computeDirectionalPreLightingInfo(float4 lightData, float3 V, float3 N) {
    PreLightingInfo result;
    result.lightDistance = length(-lightData.xyz);
    result.L = normalize(-lightData.xyz);
    result.H = normalize(V + result.L);
    result.VdotH = clamp(dot(V, result.H), 0.0, 1.0);
    result.NdotLUnclamped = dot(N, result.L);
    result.NdotL = clamp(result.NdotLUnclamped, 0.0000001, 1.0);
    return result;
}

PreLightingInfo computeHemisphericPreLightingInfo(float4 lightData, float3 V, float3 N) {
    PreLightingInfo result;
    result.NdotL = dot(N, lightData.xyz) * 0.5 + 0.5;
    result.NdotL = clamp(result.NdotL, 0.0000001, 1.0);
    result.NdotLUnclamped = result.NdotL;
    return result;
}

float computeDistanceLightFalloff_Standard(float3 lightOffset, float range) {
    return max(0., 1.0 - length(lightOffset) / range);
}

float computeDistanceLightFalloff_Physical(float lightDistanceSquared) {
    return 1.0 / max(lightDistanceSquared, 0.0000001);
}

float computeDistanceLightFalloff_GLTF(float lightDistanceSquared, float inverseSquaredRange) {
    float lightDistanceFalloff = 1.0 / max(lightDistanceSquared, 0.0000001);
    float factor = lightDistanceSquared * inverseSquaredRange;
    float attenuation = clamp(1.0 - factor * factor, 0.0, 1.0);
    attenuation *= attenuation;
    lightDistanceFalloff *= attenuation;
    return lightDistanceFalloff;
}

float computeDistanceLightFalloff(float3 lightOffset, float lightDistanceSquared, float range, float inverseSquaredRange) {
    return computeDistanceLightFalloff_Physical(lightDistanceSquared);
}

float computeDirectionalLightFalloff_Standard(float3 lightDirection, float3 directionToLightCenterW, float cosHalfAngle, float exponent) {
    float falloff = 0.0;
    float cosAngle = max(dot(-lightDirection, directionToLightCenterW), 0.0000001);
    if (cosAngle >= cosHalfAngle) {
        falloff = max(0., pow(cosAngle, exponent));
    }
    return falloff;
}

float computeDirectionalLightFalloff_Physical(float3 lightDirection, float3 directionToLightCenterW, float cosHalfAngle) {
    float kMinusLog2ConeAngleIntensityRatio = 6.64385618977;
    float concentrationKappa = kMinusLog2ConeAngleIntensityRatio / (1.0 - cosHalfAngle);
    float4 lightDirectionSpreadSG = float4(-lightDirection * concentrationKappa, -concentrationKappa);
    float falloff = exp2(dot(float4(directionToLightCenterW, 1.0), lightDirectionSpreadSG));
    return falloff;
}

float computeDirectionalLightFalloff_GLTF(float3 lightDirection, float3 directionToLightCenterW, float lightAngleScale, float lightAngleOffset) {
    float cd = dot(-lightDirection, directionToLightCenterW);
    float falloff = clamp(cd * lightAngleScale + lightAngleOffset, 0.0, 1.0);
    falloff *= falloff;
    return falloff;
}

float computeDirectionalLightFalloff(float3 lightDirection, float3 directionToLightCenterW, float cosHalfAngle, float exponent, float lightAngleScale, float lightAngleOffset) {
    return computeDirectionalLightFalloff_Physical(lightDirection, directionToLightCenterW, cosHalfAngle);
}

float3 getEnergyConservationFactor(float3 specularEnvironmentR0, float3 environmentBrdf) {
    return float3(1.0, 1.0, 1.0) + specularEnvironmentR0 * (1.0 / environmentBrdf.y - 1.0);
}

float3 getBRDFLookup(float NdotV, float perceptualRoughness, Texture2D<float4> environmentBrdfSamplerTexture, sampler environmentBrdfSamplerSampler) {
    float2 UV = float2(NdotV, perceptualRoughness);
    float4 brdfLookup = Sample(environmentBrdfSamplerTexture, environmentBrdfSamplerSampler, UV);
    return brdfLookup.xyz;
}

float3 getReflectanceFromBRDFLookup(float3 specularEnvironmentR0, float3 environmentBrdf) {
    float3 reflectance = lerp(environmentBrdf.xxx, environmentBrdf.yyy, specularEnvironmentR0);
    return reflectance;
}

float3 getReflectanceFromAnalyticalBRDFLookup_Jones(float VdotN, float3 reflectance0, float3 reflectance90, float smoothness) {
    float weight = lerp(0.25, 1.0, smoothness);
    return reflectance0 + weight * (reflectance90 - reflectance0) * pow5(clamp(1.0 - VdotN, 0.0, 1.0));
}

float3 fresnelSchlickGGX(float VdotH, float3 reflectance0, float3 reflectance90) {
    return reflectance0 + (reflectance90 - reflectance0) * pow5(1.0 - VdotH);
}

float fresnelSchlickGGX(float VdotH, float reflectance0, float reflectance90) {
    return reflectance0 + (reflectance90 - reflectance0) * pow5(1.0 - VdotH);
}

float normalDistributionFunction_TrowbridgeReitzGGX(float NdotH, float alphaG) {
    float a2 = square(alphaG);
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.1415926535897932384626433832795 * d * d);
}

float smithVisibility_GGXCorrelated(float NdotL, float NdotV, float alphaG) {
    float a2 = alphaG * alphaG;
    float GGXV = NdotL * sqrt(NdotV * (NdotV - a2 * NdotV) + a2);
    float GGXL = NdotV * sqrt(NdotL * (NdotL - a2 * NdotL) + a2);
    return 0.5 / (GGXV + GGXL);
}

float diffuseBRDF_Burley(float NdotL, float NdotV, float VdotH, float roughness) {
    float diffuseFresnelNV = pow5(clamp(1.0 - NdotL, 0.0000001, 1.0));
    float diffuseFresnelNL = pow5(clamp(1.0 - NdotV, 0.0000001, 1.0));
    float diffuseFresnel90 = 0.5 + 2.0 * VdotH * VdotH * roughness;
    float fresnel = (1.0 + (diffuseFresnel90 - 1.0) * diffuseFresnelNL) *
        (1.0 + (diffuseFresnel90 - 1.0) * diffuseFresnelNV);
    return fresnel / 3.1415926535897932384626433832795;
}

struct LightingInfo {
    float3 diffuse;
}

float adjustRoughnessFromLightProperties(float roughness, float lightRadius, float lightDistance) {
    float lightRoughness = lightRadius / lightDistance;
    float totalRoughness = clamp(lightRoughness + roughness, 0.0, 1.0);
    return totalRoughness;
}

float3 computeHemisphericDiffuseLighting(PreLightingInfo info, float3 lightColor, float3 groundColor) {
    return lerp(groundColor, lightColor, float3(info.NdotL, info.NdotL, info.NdotL));
}

float3 computeDiffuseLighting(PreLightingInfo info, float3 lightColor) {
    float diffuseTerm = diffuseBRDF_Burley(info.NdotL, info.NdotV, info.VdotH, info.roughness);
    return diffuseTerm * info.attenuation * info.NdotL * lightColor;
}

float2 computeProjectionTextureDiffuseLightingUV(float4x4 textureProjectionMatrix, float3 vPositionW) {
    float4 strq = mul(textureProjectionMatrix, float4(vPositionW, 1.0));
    strq /= strq.w;
    return strq.xy;
}

float getLodFromAlphaG(float cubeMapDimensionPixels, float microsurfaceAverageSlope) {
    float microsurfaceAverageSlopeTexels = cubeMapDimensionPixels * microsurfaceAverageSlope;
    float lod = log2(microsurfaceAverageSlopeTexels);
    return lod;
}

float getLinearLodFromRoughness(float cubeMapDimensionPixels, float roughness) {
    float lod = log2(cubeMapDimensionPixels) * roughness;
    return lod;
}

float environmentRadianceOcclusion(float ambientOcclusion, float NdotVUnclamped) {
    float temp = NdotVUnclamped + ambientOcclusion;
    return clamp(square(temp) - 1.0 + ambientOcclusion, 0.0, 1.0);
}

float environmentHorizonOcclusion(float3 view, float3 normal) {
    float3 reflection = reflect(view, normal);
    float temp = clamp(1.0 + 1.1 * dot(reflection, normal), 0.0, 1.0);
    return square(temp);
}

float3 parallaxCorrectNormal(float3 vertexPos, float3 origVec, float3 cubeSize, float3 cubePos) {
    float3 invOrigVec = float3(1.0, 1.0, 1.0) / origVec;
    float3 halfSize = cubeSize * 0.5;
    float3 intersecAtMaxPlane = (cubePos + halfSize - vertexPos) * invOrigVec;
    float3 intersecAtMinPlane = (cubePos - halfSize - vertexPos) * invOrigVec;
    float3 largestIntersec = max(intersecAtMaxPlane, intersecAtMinPlane);
    float distance = min(min(largestIntersec.x, largestIntersec.y), largestIntersec.z);
    float3 intersectPositionWS = vertexPos + origVec * distance;
    return intersectPositionWS - cubePos;
}

float3 computeFixedEquirectangularCoords(float4 worldPos, float3 worldNormal, float3 direction) {
    float lon = atan2(direction.z, direction.x);
    float lat = acos(direction.y);
    float2 sphereCoords = float2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return float3(s, t, 0);
}

float3 computeMirroredFixedEquirectangularCoords(float4 worldPos, float3 worldNormal, float3 direction) {
    float lon = atan2(direction.z, direction.x);
    float lat = acos(direction.y);
    float2 sphereCoords = float2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return float3(1.0 - s, t, 0);
}

float3 computeEquirectangularCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix) {
    float3 cameraToVertex = normalize(worldPos.xyz - eyePosition);
    float3 r = normalize(reflect(cameraToVertex, worldNormal));
    r = mul(reflectionMatrix, float4(r, 0)).xyz;
    float lon = atan2(r.z, r.x);
    float lat = acos(r.y);
    float2 sphereCoords = float2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return float3(s, t, 0);
}

float3 computeSphericalCoords(float4 worldPos, float3 worldNormal, float4x4 view, float4x4 reflectionMatrix) {
    float3 viewDir = normalize(mul(view, worldPos).xyz);
    float3 viewNormal = normalize(mul(view, float4(worldNormal, 0.0)).xyz);
    float3 r = reflect(viewDir, viewNormal);
    r = mul(reflectionMatrix, float4(r, 0)).xyz;
    r.z = r.z - 1.0;
    float m = 2.0 * length(r);
    return float3(r.x / m + 0.5, 1.0 - r.y / m - 0.5, 0);
}

float3 computePlanarCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix) {
    float3 viewDir = worldPos.xyz - eyePosition;
    float3 coords = normalize(reflect(viewDir, worldNormal));
    return mul(reflectionMatrix, float4(coords, 1)).xyz;
}

float3 computeCubicCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix) {
    float3 viewDir = normalize(worldPos.xyz - eyePosition);
    float3 coords = reflect(viewDir, worldNormal);
    coords = mul(reflectionMatrix, float4(coords, 0)).xyz;
    return coords;
}

float3 computeCubicLocalCoords(float4 worldPos, float3 worldNormal, float3 eyePosition, float4x4 reflectionMatrix, float3 reflectionSize, float3 reflectionPosition) {
    float3 viewDir = normalize(worldPos.xyz - eyePosition);
    float3 coords = reflect(viewDir, worldNormal);
    coords = parallaxCorrectNormal(worldPos.xyz, coords, reflectionSize, reflectionPosition);
    coords = mul(reflectionMatrix, float4(coords, 0)).xyz;
    return coords;
}

float3 computeProjectionCoords(float4 worldPos, float4x4 view, float4x4 reflectionMatrix) {
    return mul(reflectionMatrix, mul(view, worldPos)).xyz;
}

float3 computeSkyBoxCoords(float3 positionW, float4x4 reflectionMatrix) {
    return mul(reflectionMatrix, float4(positionW, 0)).xyz;
}

float3 computeReflectionCoords(float4 worldPos, float3 worldNormal, float3 vPositionUVW, float4x4 reflectionMatrix) {
    float r = length(vPositionUVW);
    float gamma = atan2(vPositionUVW.x, vPositionUVW.z);
    float theta = acos(vPositionUVW.y / r);
    return float3(gamma / 3.1415926535897932384626433832795 + 0.5, theta / 3.1415926535897932384626433832795, r);
}

fragment float4 main(float3 vPositionW : attribute(0), float3 vNormalW : attribute(1), float3 vPositionUVW : attribute(2),
        constant Scene[] scene : register(b0, space0),
        Texture2D<float4> environmentBrdfSamplerTexture : register(t1, space0),
        sampler environmentBrdfSamplerSampler : register(s2, space0),
        constant Material[] material : register(b0, space1),
        constant Mesh[] mesh : register(b1, space1),
        Texture2D<float4> reflectionSamplerTexture : register(t0, space2),
        sampler reflectionSamplerSampler : register(s1, space2),
        bool frontFace : SV_IsFrontFace) : SV_Target 0 {
    float3 viewDirectionW = normalize(material[0].vEyePosition.xyz - vPositionW);
    float3 normalW = normalize(vNormalW);
    float2 uvOffset = float2(0.0, 0.0);
    normalW = frontFace ? normalW : -normalW;
    float3 surfaceAlbedo = material[0].vAlbedoColor.xyz;
    float alpha = material[0].vAlbedoColor.w;
    float3 ambientOcclusionColor = float3(1., 1., 1.);
    float microSurface = material[0].vReflectivityColor.w;
    float3 surfaceReflectivityColor = material[0].vReflectivityColor.xyz;
    microSurface = clamp(microSurface, 0.0, 1.0);
    float roughness = 1. - microSurface;
    float NdotVUnclamped = dot(normalW, viewDirectionW);
    float NdotV = abs(NdotVUnclamped) + 0.0000001;
    float alphaG = convertRoughnessToAverageSlope(roughness);
    float2 AARoughnessFactors = getAARoughnessFactors(normalW);
    float4 environmentRadiance = float4(0., 0., 0., 0.);
    float3 environmentIrradiance = float3(0., 0., 0.);
    float3 reflectionVector = computeReflectionCoords(float4(vPositionW, 1.0), normalW, vPositionUVW, material[0].reflectionMatrix);
    float3 reflectionCoords = reflectionVector;
    float reflectionLOD = getLodFromAlphaG(material[0].vReflectionMicrosurfaceInfos.x, alphaG);
    reflectionLOD = reflectionLOD * material[0].vReflectionMicrosurfaceInfos.y + material[0].vReflectionMicrosurfaceInfos.z;
    float requestedReflectionLOD = reflectionLOD;
    environmentRadiance = Sample(reflectionSamplerTexture, reflectionSamplerSampler, reflectionCoords.xy); //SampleLevel(reflectionSamplerTexture, reflectionSamplerSampler, reflectionCoords, requestedReflectionLOD);
    environmentRadiance.xyz = fromRGBD(environmentRadiance);
    environmentRadiance.xyz *= material[0].vReflectionInfos.x;
    environmentRadiance.xyz *= material[0].vReflectionColor.xyz;
    environmentIrradiance *= material[0].vReflectionColor.xyz;
    float reflectance = max(max(surfaceReflectivityColor.x, surfaceReflectivityColor.y), surfaceReflectivityColor.z);
    float reflectance90 = fresnelGrazingReflectance(reflectance);
    float3 specularEnvironmentR0 = surfaceReflectivityColor.xyz;
    float3 specularEnvironmentR90 = float3(1.0, 1.0, 1.0) * reflectance90;
    float3 environmentBrdf = getBRDFLookup(NdotV, roughness, environmentBrdfSamplerTexture, environmentBrdfSamplerSampler);
    float3 energyConservationFactor = getEnergyConservationFactor(specularEnvironmentR0, environmentBrdf);
    float3 diffuseBase = float3(0., 0., 0.);
    PreLightingInfo preInfo;
    LightingInfo info;
    float shadow = 1.;
    float3 specularEnvironmentReflectance = getReflectanceFromAnalyticalBRDFLookup_Jones(NdotV, specularEnvironmentR0, specularEnvironmentR90, sqrt(microSurface));
    surfaceAlbedo.xyz = (1. - reflectance) * surfaceAlbedo.xyz;
    float3 finalIrradiance = environmentIrradiance;
    finalIrradiance *= surfaceAlbedo.xyz;
    float3 finalRadiance = environmentRadiance.xyz;
    finalRadiance *= specularEnvironmentReflectance;
    float3 finalRadianceScaled = finalRadiance * material[0].vLightingIntensity.z;
    finalRadianceScaled *= energyConservationFactor;
    float3 finalDiffuse = diffuseBase;
    finalDiffuse *= surfaceAlbedo.xyz;
    finalDiffuse = max(finalDiffuse, float3(0.0, 0.0, 0.0));
    float3 finalAmbient = material[0].vAmbientColor;
    finalAmbient *= surfaceAlbedo.xyz;
    float3 finalEmissive = material[0].vEmissiveColor;
    float3 ambientOcclusionForDirectDiffuse = ambientOcclusionColor;
    float4 finalColor = float4(
        finalAmbient * ambientOcclusionColor +
        finalDiffuse * ambientOcclusionForDirectDiffuse * material[0].vLightingIntensity.x +
        finalIrradiance * ambientOcclusionColor * material[0].vLightingIntensity.z +
        finalRadianceScaled +
        finalEmissive * material[0].vLightingIntensity.y,
        alpha);
    finalColor = max(finalColor, float4(0.0, 0.0, 0.0, 0.0));
    finalColor = applyImageProcessing(finalColor);
    finalColor.w *= mesh[0].visibility;
    return finalColor;
}
`;