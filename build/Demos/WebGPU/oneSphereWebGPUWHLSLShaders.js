const vertexShaderWHLSL1 = `
struct Output {
    float3 vPositionW : attribute(0);
    float3 vNormalW : attribute(1);
    float3 vEnvironmentIrradiance : attribute(2);
    float4 position : SV_Position;
}

struct Scene {
    float4x4 viewProjection;
    float4x4 view;
}

struct BindGroupA {
    constant Scene[] scene : register(b0, space0);
    Texture2D<float> aTexture : register(t1, space0);
    sampler aSampler : register(s2, space0);
}

struct Mesh {
    float4x4 world;
    float visibility;
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
};

struct BindGroupB {
    constant Material[] material : register(b0, space1);
    constant Mesh[] mesh : register(b1, space1);
}

struct BindGroupC {
    Texture2D<float> aTexture : register(t0, space2);
    sampler aSampler : register(s1, space2);
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

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1), BindGroupA bindGroupA, BindGroupB bindGroupB, BindGroupC bindGroupC) {
    Output output;

    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    float4x4 finalWorld = bindGroupB.mesh[0].world;
    output.position = mul(mul(bindGroupA.scene[0].viewProjection, finalWorld), float4(positionUpdated, 1.0));
    float4 worldPos = mul(finalWorld, float4(positionUpdated, 1.0));
    output.vPositionW = worldPos.xyz;
    float3x3 normalWorld;
    normalWorld[0] = finalWorld[0].xyz;
    normalWorld[1] = finalWorld[1].xyz;
    normalWorld[2] = finalWorld[2].xyz;
    output.vNormalW = normalize(mul(normalWorld, normalUpdated));
    float3 reflectionVector = mul(bindGroupB.material[0].reflectionMatrix, float4(output.vNormalW, 0)).xyz;
    output.vEnvironmentIrradiance = computeEnvironmentIrradiance(reflectionVector, &bindGroupB.material[0]);
    float2 uvUpdated;
    float2 uv2;
    //output.position *= -1.;
    return output;
}
`;

const fragmentShaderWHLSL1 = `
fragment float4 main(float3 vPositionW : attribute(0), float3 vNormalW : attribute(1), float3 vEnvironmentIrradiance : attribute(2)) : SV_Target 0 {
    return float4(1, 0, 0, 1);
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

struct BindGroupA {
    constant Scene[] scene : register(b0, space0);
    Texture2D<float> aTexture : register(t1, space0);
    sampler aSampler : register(s2, space0);
}

struct Mesh {
    float4x4 world;
    float visibility;
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
};

struct BindGroupB {
    constant Material[] aBuffer : register(b0, space1);
    constant Mesh[] mesh : register(b1, space1);
}

struct BindGroupC {
    Texture2D<float> aTexture : register(t0, space2);
    sampler aSampler : register(s1, space2);
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

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1), BindGroupA bindGroupA, BindGroupB bindGroupB, BindGroupC bindGroupC) {
    Output output;
    
    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    output.vPositionUVW = positionUpdated;
    float4x4 finalWorld = bindGroupB.mesh[0].world;
    output.position = mul(mul(bindGroupA.scene[0].viewProjection, finalWorld), float4(positionUpdated, 1.0));
    float4 worldPos = mul(finalWorld, float4(positionUpdated, 1.0));
    output.vPositionW = worldPos.xyz;
    float3x3 normalWorld;
    normalWorld[0] = finalWorld[0].xyz;
    normalWorld[1] = finalWorld[1].xyz;
    normalWorld[2] = finalWorld[2].xyz;
    output.vNormalW = normalize(mul(normalWorld, normalUpdated));
    float2 uvUpdated;
    float2 uv2;
    //output.position *= -1.;
    return output;
}
`;

const fragmentShaderWHLSL2 = `
fragment float4 main(float3 vPositionW : attribute(0), float3 vNormalW : attribute(1), float3 vPositionUVW : attribute(2)) : SV_Target 0 {
    return float4(0, 1, 0, 1);
}
`;
