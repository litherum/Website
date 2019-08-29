#version 450

precision highp float;

layout(std140, column_major) uniform;

layout(set = 0, binding = 0) uniform Scene {
    mat4 viewProjection;
    mat4 view;
};

layout(set = 1, binding = 0) uniform Material {
    vec2 vAlbedoInfos;
    vec4 vAmbientInfos;
    vec2 vOpacityInfos;
    vec2 vEmissiveInfos;
    vec2 vLightmapInfos;
    vec3 vReflectivityInfos;
    vec2 vMicroSurfaceSamplerInfos;
    vec2 vReflectionInfos;
    vec3 vReflectionPosition;
    vec3 vReflectionSize;
    vec3 vBumpInfos;
    mat4 albedoMatrix;
    mat4 ambientMatrix;
    mat4 opacityMatrix;
    mat4 emissiveMatrix;
    mat4 lightmapMatrix;
    mat4 reflectivityMatrix;
    mat4 microSurfaceSamplerMatrix;
    mat4 bumpMatrix;
    vec2 vTangentSpaceParams;
    mat4 reflectionMatrix;
    vec3 vReflectionColor;
    vec4 vAlbedoColor;
    vec4 vLightingIntensity;
    vec3 vReflectionMicrosurfaceInfos;
    float pointSize;
    vec4 vReflectivityColor;
    vec3 vEmissiveColor;
    vec4 vEyePosition;
    vec3 vAmbientColor;
    vec2 vDebugMode;
    vec2 vClearCoatParams;
    vec4 vClearCoatRefractionParams;
    vec2 vClearCoatInfos;
    mat4 clearCoatMatrix;
    vec2 vClearCoatBumpInfos;
    vec2 vClearCoatTangentSpaceParams;
    mat4 clearCoatBumpMatrix;
    vec4 vClearCoatTintParams;
    float clearCoatColorAtDistance;
    vec2 vClearCoatTintInfos;
    mat4 clearCoatTintMatrix;
    vec3 vAnisotropy;
    vec2 vAnisotropyInfos;
    mat4 anisotropyMatrix;
    vec4 vSheenColor;
    vec2 vSheenInfos;
    mat4 sheenMatrix;
    vec3 vRefractionMicrosurfaceInfos;
    vec4 vRefractionInfos;
    mat4 refractionMatrix;
    vec2 vThicknessInfos;
    mat4 thicknessMatrix;
    vec2 vThicknessParam;
    vec3 vDiffusionDistance;
    vec4 vTintColor;
    vec3 vSubSurfaceIntensity;
    vec3 vSphericalL00;
    vec3 vSphericalL1_1;
    vec3 vSphericalL10;
    vec3 vSphericalL11;
    vec3 vSphericalL2_2;
    vec3 vSphericalL2_1;
    vec3 vSphericalL20;
    vec3 vSphericalL21;
    vec3 vSphericalL22;
    vec3 vSphericalX;
    vec3 vSphericalY;
    vec3 vSphericalZ;
    vec3 vSphericalXX_ZZ;
    vec3 vSphericalYY_ZZ;
    vec3 vSphericalZZ;
    vec3 vSphericalXY;
    vec3 vSphericalYZ;
    vec3 vSphericalZX;
};

layout(set = 1, binding = 1) uniform Mesh {
    mat4 world;
    float visibility;
};

layout(location = 0) in vec3 vPositionW;
layout(location = 1) in vec3 vNormalW;
layout(set = 0, binding = 1) uniform texture2D environmentBrdfSamplerTexture;
layout(set = 0, binding = 2) uniform sampler environmentBrdfSamplerSampler;
layout(set = 2, binding = 0) uniform textureCube reflectionSamplerTexture;
layout(set = 2, binding = 1) uniform sampler reflectionSamplerSampler;
layout(location = 2) in vec3 vPositionUVW;

mat3 transposeMat3(mat3 inMatrix) {
    vec3 i0 = inMatrix[0];
    vec3 i1 = inMatrix[1];
    vec3 i2 = inMatrix[2];
    mat3 outMatrix = mat3(
        vec3(i0.x, i1.x, i2.x),
        vec3(i0.y, i1.y, i2.y),
        vec3(i0.z, i1.z, i2.z)
    );
    return outMatrix;
}

mat3 inverseMat3(mat3 inMatrix) {
    float a00 = inMatrix[0][0], a01 = inMatrix[0][1], a02 = inMatrix[0][2];
    float a10 = inMatrix[1][0], a11 = inMatrix[1][1], a12 = inMatrix[1][2];
    float a20 = inMatrix[2][0], a21 = inMatrix[2][1], a22 = inMatrix[2][2];
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
    float det = a00 * b01 + a01 * b11 + a02 * b21;
    return mat3(b01, -a22 * a01 + a02 * a21, a12 * a01 - a02 * a11,
        b11, a22 * a00 - a02 * a20, -a12 * a00 + a02 * a10,
        b21, -a21 * a00 + a01 * a20, a11 * a00 - a01 * a10) / det;
}

vec3 toLinearSpace(vec3 color) {
    return pow(color, vec3(2.2));
}

vec3 toGammaSpace(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

float square(float value) {
    return value * value;
}

float pow5(float value) {
    float sq = value * value;
    return sq * sq * value;
}

float getLuminance(vec3 color) {
    return clamp(dot(color, vec3(0.2126, 0.7152, 0.0722)), 0., 1.);
}

float getRand(vec2 seed) {
    return fract(sin(dot(seed.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float dither(vec2 seed, float varianceAmount) {
    float rand = getRand(seed);
    float dither = mix(-varianceAmount / 255.0, varianceAmount / 255.0, rand);
    return dither;
}

vec4 toRGBD(vec3 color) {
    float maxRGB = max(max(color.r, max(color.g, color.b)), 0.0000001);
    float D = max(255.0 / maxRGB, 1.);
    D = clamp(floor(D) / 255.0, 0., 1.);
    vec3 rgb = color.rgb * D;
    rgb = toGammaSpace(rgb);
    return vec4(rgb, D);
}

vec3 fromRGBD(vec4 rgbd) {
    rgbd.rgb = toLinearSpace(rgbd.rgb);
    return rgbd.rgb / rgbd.a;
}

float convertRoughnessToAverageSlope(float roughness) {
    return square(roughness) + 0.0005;
}

float fresnelGrazingReflectance(float reflectance0) {
    float reflectance90 = clamp(reflectance0 * 25.0, 0.0, 1.0);
    return reflectance90;
}

vec2 getAARoughnessFactors(vec3 normalVector) {
    return vec2(0.);
}

vec4 applyImageProcessing(vec4 result) {
    result.rgb = toGammaSpace(result.rgb);
    result.rgb = clamp(result.rgb, 0.0, 1.0);
    return result;
}

struct preLightingInfo {
    vec3 lightOffset;
    float lightDistanceSquared;
    float lightDistance;
    float attenuation;
    vec3 L;
    vec3 H;
    float NdotV;
    float NdotLUnclamped;
    float NdotL;
    float VdotH;
    float roughness;
};

preLightingInfo computePointAndSpotPreLightingInfo(vec4 lightData, vec3 V, vec3 N) {
    preLightingInfo result;
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

preLightingInfo computeDirectionalPreLightingInfo(vec4 lightData, vec3 V, vec3 N) {
    preLightingInfo result;
    result.lightDistance = length(-lightData.xyz);
    result.L = normalize(-lightData.xyz);
    result.H = normalize(V + result.L);
    result.VdotH = clamp(dot(V, result.H), 0.0, 1.0);
    result.NdotLUnclamped = dot(N, result.L);
    result.NdotL = clamp(result.NdotLUnclamped, 0.0000001, 1.0);
    return result;
}

preLightingInfo computeHemisphericPreLightingInfo(vec4 lightData, vec3 V, vec3 N) {
    preLightingInfo result;
    result.NdotL = dot(N, lightData.xyz) * 0.5+0.5;
    result.NdotL = clamp(result.NdotL, 0.0000001, 1.0);
    result.NdotLUnclamped = result.NdotL;
    return result;
}

float computeDistanceLightFalloff_Standard(vec3 lightOffset, float range) {
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

float computeDistanceLightFalloff(vec3 lightOffset, float lightDistanceSquared, float range, float inverseSquaredRange) {
    return computeDistanceLightFalloff_Physical(lightDistanceSquared);
}

float computeDirectionalLightFalloff_Standard(vec3 lightDirection, vec3 directionToLightCenterW, float cosHalfAngle, float exponent) {
    float falloff = 0.0;
    float cosAngle = max(dot(-lightDirection, directionToLightCenterW), 0.0000001);
    if (cosAngle >= cosHalfAngle)
        falloff = max(0., pow(cosAngle, exponent));
    return falloff;
}

float computeDirectionalLightFalloff_Physical(vec3 lightDirection, vec3 directionToLightCenterW, float cosHalfAngle)
{
    const float kMinusLog2ConeAngleIntensityRatio = 6.64385618977;
    float concentrationKappa = kMinusLog2ConeAngleIntensityRatio / (1.0 - cosHalfAngle);
    vec4 lightDirectionSpreadSG = vec4(-lightDirection * concentrationKappa, -concentrationKappa);
    float falloff = exp2(dot(vec4(directionToLightCenterW, 1.0), lightDirectionSpreadSG));
    return falloff;
}

float computeDirectionalLightFalloff_GLTF(vec3 lightDirection, vec3 directionToLightCenterW, float lightAngleScale, float lightAngleOffset) {
    float cd = dot(-lightDirection, directionToLightCenterW);
    float falloff = clamp(cd * lightAngleScale + lightAngleOffset, 0.0, 1.0);
    falloff *= falloff;
    return falloff;
}

float computeDirectionalLightFalloff(vec3 lightDirection, vec3 directionToLightCenterW, float cosHalfAngle, float exponent, float lightAngleScale, float lightAngleOffset) {
    return computeDirectionalLightFalloff_Physical(lightDirection, directionToLightCenterW, cosHalfAngle);
}

vec3 getEnergyConservationFactor(const vec3 specularEnvironmentR0, const vec3 environmentBrdf) {
    return 1.0 + specularEnvironmentR0 * (1.0 / environmentBrdf.y - 1.0);
}

vec3 getBRDFLookup(float NdotV, float perceptualRoughness) {
    vec2 UV = vec2(NdotV, perceptualRoughness);
    vec4 brdfLookup = texture(sampler2D(environmentBrdfSamplerTexture, environmentBrdfSamplerSampler), UV);
    return brdfLookup.rgb;
}

vec3 getReflectanceFromBRDFLookup(const vec3 specularEnvironmentR0, const vec3 environmentBrdf) {
    vec3 reflectance = mix(environmentBrdf.xxx, environmentBrdf.yyy, specularEnvironmentR0);
    return reflectance;
}

vec3 getReflectanceFromAnalyticalBRDFLookup_Jones(float VdotN, vec3 reflectance0, vec3 reflectance90, float smoothness) {
    float weight = mix(0.25, 1.0, smoothness);
    return reflectance0 + weight * (reflectance90 - reflectance0) * pow5(clamp(1.0 - VdotN, 0.0, 1.0));
}

vec3 fresnelSchlickGGX(float VdotH, vec3 reflectance0, vec3 reflectance90) {
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
    float fresnel =
        (1.0 + (diffuseFresnel90 - 1.0) * diffuseFresnelNL) *
        (1.0 + (diffuseFresnel90 - 1.0) * diffuseFresnelNV);
    return fresnel / 3.1415926535897932384626433832795;
}

struct lightingInfo {
    vec3 diffuse;
};

float adjustRoughnessFromLightProperties(float roughness, float lightRadius, float lightDistance) {
    float lightRoughness = lightRadius / lightDistance;
    float totalRoughness = clamp(lightRoughness + roughness, 0.0, 1.0);
    return totalRoughness;
}

vec3 computeHemisphericDiffuseLighting(preLightingInfo info, vec3 lightColor, vec3 groundColor) {
    return mix(groundColor, lightColor, info.NdotL);
}

vec3 computeDiffuseLighting(preLightingInfo info, vec3 lightColor) {
    float diffuseTerm = diffuseBRDF_Burley(info.NdotL, info.NdotV, info.VdotH, info.roughness);
    return diffuseTerm * info.attenuation * info.NdotL * lightColor;
}

vec2 computeProjectionTextureDiffuseLightingUV(mat4 textureProjectionMatrix) {
    vec4 strq = textureProjectionMatrix * vec4(vPositionW, 1.0);
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

float environmentHorizonOcclusion(vec3 view, vec3 normal) {
    vec3 reflection = reflect(view, normal);
    float temp = clamp(1.0 + 1.1 * dot(reflection, normal), 0.0, 1.0);
    return square(temp);
}

vec3 parallaxCorrectNormal(vec3 vertexPos, vec3 origVec, vec3 cubeSize, vec3 cubePos) {
    vec3 invOrigVec = vec3(1.0, 1.0, 1.0) / origVec;
    vec3 halfSize = cubeSize * 0.5;
    vec3 intersecAtMaxPlane = (cubePos + halfSize - vertexPos) * invOrigVec;
    vec3 intersecAtMinPlane = (cubePos - halfSize - vertexPos) * invOrigVec;
    vec3 largestIntersec = max(intersecAtMaxPlane, intersecAtMinPlane);
    float distance = min(min(largestIntersec.x, largestIntersec.y), largestIntersec.z);
    vec3 intersectPositionWS = vertexPos + origVec * distance;
    return intersectPositionWS - cubePos;
}

vec3 computeFixedEquirectangularCoords(vec4 worldPos, vec3 worldNormal, vec3 direction) {
    float lon = atan(direction.z, direction.x);
    float lat = acos(direction.y);
    vec2 sphereCoords = vec2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return vec3(s, t, 0);
}

vec3 computeMirroredFixedEquirectangularCoords(vec4 worldPos, vec3 worldNormal, vec3 direction) {
    float lon = atan(direction.z, direction.x);
    float lat = acos(direction.y);
    vec2 sphereCoords = vec2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return vec3(1.0 - s, t, 0);
}

vec3 computeEquirectangularCoords(vec4 worldPos, vec3 worldNormal, vec3 eyePosition, mat4 reflectionMatrix) {
    vec3 cameraToVertex = normalize(worldPos.xyz - eyePosition);
    vec3 r = normalize(reflect(cameraToVertex, worldNormal));
    r = vec3(reflectionMatrix * vec4(r, 0));
    float lon = atan(r.z, r.x);
    float lat = acos(r.y);
    vec2 sphereCoords = vec2(lon, lat) * 0.15915494 * 2.0;
    float s = sphereCoords.x * 0.5 + 0.5;
    float t = sphereCoords.y;
    return vec3(s, t, 0);
}

vec3 computeSphericalCoords(vec4 worldPos, vec3 worldNormal, mat4 view, mat4 reflectionMatrix) {
    vec3 viewDir = normalize(vec3(view * worldPos));
    vec3 viewNormal = normalize(vec3(view * vec4(worldNormal, 0.0)));
    vec3 r = reflect(viewDir, viewNormal);
    r = vec3(reflectionMatrix * vec4(r, 0));
    r.z = r.z - 1.0;
    float m = 2.0 * length(r);
    return vec3(r.x / m + 0.5, 1.0 - r.y / m - 0.5, 0);
}

vec3 computePlanarCoords(vec4 worldPos, vec3 worldNormal, vec3 eyePosition, mat4 reflectionMatrix) {
    vec3 viewDir = worldPos.xyz - eyePosition;
    vec3 coords = normalize(reflect(viewDir, worldNormal));
    return vec3(reflectionMatrix * vec4(coords, 1));
}

vec3 computeCubicCoords(vec4 worldPos, vec3 worldNormal, vec3 eyePosition, mat4 reflectionMatrix) {
    vec3 viewDir = normalize(worldPos.xyz - eyePosition);
    vec3 coords = reflect(viewDir, worldNormal);
    coords = vec3(reflectionMatrix * vec4(coords, 0));
    return coords;
}

vec3 computeCubicLocalCoords(vec4 worldPos, vec3 worldNormal, vec3 eyePosition, mat4 reflectionMatrix, vec3 reflectionSize, vec3 reflectionPosition) {
    vec3 viewDir = normalize(worldPos.xyz - eyePosition);
    vec3 coords = reflect(viewDir, worldNormal);
    coords = parallaxCorrectNormal(worldPos.xyz, coords, reflectionSize, reflectionPosition);
    coords = vec3(reflectionMatrix * vec4(coords, 0));
    return coords;
}

vec3 computeProjectionCoords(vec4 worldPos, mat4 view, mat4 reflectionMatrix) {
    return vec3(reflectionMatrix * (view * worldPos));
}

vec3 computeSkyBoxCoords(vec3 positionW, mat4 reflectionMatrix) {
    return vec3(reflectionMatrix * vec4(positionW, 0));
}

vec3 computeReflectionCoords(vec4 worldPos, vec3 worldNormal) {
    return computeSkyBoxCoords(vPositionUVW, reflectionMatrix);
}

layout(location = 0) out vec4 glFragColor;

void main(void) {
    vec3 viewDirectionW = normalize(vEyePosition.xyz - vPositionW);
    vec3 normalW = normalize(vNormalW);
    vec2 uvOffset = vec2(0.0, 0.0);
    normalW = gl_FrontFacing ? normalW : -normalW;
    vec3 surfaceAlbedo = vAlbedoColor.rgb;
    float alpha = vAlbedoColor.a;
    vec3 ambientOcclusionColor = vec3(1., 1., 1.);
    float microSurface = vReflectivityColor.a;
    vec3 surfaceReflectivityColor = vReflectivityColor.rgb;
    microSurface = clamp(microSurface, 0.0, 1.0);
    float roughness = 1. - microSurface;
    float NdotVUnclamped = dot(normalW, viewDirectionW);
    float NdotV = abs(NdotVUnclamped) + 0.0000001;
    float alphaG = convertRoughnessToAverageSlope(roughness);
    vec2 AARoughnessFactors = getAARoughnessFactors(normalW.xyz);
    vec4 environmentRadiance = vec4(0., 0., 0., 0.);
    vec3 environmentIrradiance = vec3(0., 0., 0.);
    vec3 reflectionVector = computeReflectionCoords(vec4(vPositionW, 1.0), normalW);
    vec3 reflectionCoords = reflectionVector;
    float reflectionLOD = getLodFromAlphaG(vReflectionMicrosurfaceInfos.x, alphaG);
    reflectionLOD = reflectionLOD * vReflectionMicrosurfaceInfos.y + vReflectionMicrosurfaceInfos.z;
    float requestedReflectionLOD = reflectionLOD;
    environmentRadiance = textureLod(samplerCube(reflectionSamplerTexture, reflectionSamplerSampler), reflectionCoords, requestedReflectionLOD);
    environmentRadiance.rgb = fromRGBD(environmentRadiance);
    environmentRadiance.rgb *= vReflectionInfos.x;
    environmentRadiance.rgb *= vReflectionColor.rgb;
    environmentIrradiance *= vReflectionColor.rgb;
    float reflectance = max(max(surfaceReflectivityColor.r, surfaceReflectivityColor.g), surfaceReflectivityColor.b);
    float reflectance90 = fresnelGrazingReflectance(reflectance);
    vec3 specularEnvironmentR0 = surfaceReflectivityColor.rgb;
    vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;
    vec3 environmentBrdf = getBRDFLookup(NdotV, roughness);
    vec3 energyConservationFactor = getEnergyConservationFactor(specularEnvironmentR0, environmentBrdf);
    vec3 diffuseBase = vec3(0., 0., 0.);
    preLightingInfo preInfo;
    lightingInfo info;
    float shadow = 1.;
    vec3 specularEnvironmentReflectance = getReflectanceFromAnalyticalBRDFLookup_Jones(NdotV, specularEnvironmentR0, specularEnvironmentR90, sqrt(microSurface));
    surfaceAlbedo.rgb = (1. - reflectance) * surfaceAlbedo.rgb;
    vec3 finalIrradiance = environmentIrradiance;
    finalIrradiance *= surfaceAlbedo.rgb;
    vec3 finalRadiance = environmentRadiance.rgb;
    finalRadiance *= specularEnvironmentReflectance;
    vec3 finalRadianceScaled = finalRadiance * vLightingIntensity.z;
    finalRadianceScaled *= energyConservationFactor;
    vec3 finalDiffuse = diffuseBase;
    finalDiffuse *= surfaceAlbedo.rgb;
    finalDiffuse = max(finalDiffuse, 0.0);
    vec3 finalAmbient = vAmbientColor;
    finalAmbient *= surfaceAlbedo.rgb;
    vec3 finalEmissive = vEmissiveColor;
    vec3 ambientOcclusionForDirectDiffuse = ambientOcclusionColor;
    vec4 finalColor = vec4(
        finalAmbient * ambientOcclusionColor +
        finalDiffuse * ambientOcclusionForDirectDiffuse * vLightingIntensity.x +
        finalIrradiance * ambientOcclusionColor * vLightingIntensity.z +
        finalRadianceScaled +
        finalEmissive * vLightingIntensity.y,
        alpha);
    finalColor = max(finalColor, 0.0);
    finalColor = applyImageProcessing(finalColor);
    finalColor.a *= visibility;
    glFragColor = finalColor;
}

