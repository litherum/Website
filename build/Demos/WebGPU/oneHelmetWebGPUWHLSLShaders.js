const vertexShaderWHLSL1 = `
#version 450
#define PBR
#define MAINUV1
#define UV1
#define ALBEDO
#define ALBEDODIRECTUV 1
#define AMBIENT
#define AMBIENTDIRECTUV 1
#define AMBIENTINGRAYSCALE
#define OPACITYDIRECTUV 0
#define ALPHATESTVALUE 0.4
#define SPECULAROVERALPHA
#define RADIANCEOVERALPHA
#define EMISSIVE
#define EMISSIVEDIRECTUV 1
#define REFLECTIVITY
#define REFLECTIVITYDIRECTUV 1
#define LODBASEDMICROSFURACE
#define MICROSURFACEMAPDIRECTUV 0
#define METALLICWORKFLOW
#define ROUGHNESSSTOREINMETALMAPGREEN
#define METALLNESSSTOREINMETALMAPBLUE
#define ENVIRONMENTBRDF
#define NORMAL
#define BUMP
#define BUMPDIRECTUV 1
#define NORMALXYSCALE
#define LIGHTMAPDIRECTUV 0
#define REFLECTION
#define REFLECTIONMAP_3D
#define REFLECTIONMAP_CUBIC
#define USESPHERICALFROMREFLECTIONMAP
#define SPHERICAL_HARMONICS
#define RGBDREFLECTION
#define RADIANCEOCCLUSION
#define HORIZONOCCLUSION
#define NUM_BONE_INFLUENCERS 0
#define BonesPerMesh 0
#define NUM_MORPH_INFLUENCERS 0
#define VIGNETTEBLENDMODEMULTIPLY
#define SAMPLER3DGREENDEPTH
#define SAMPLER3DBGRMAP
#define USEPHYSICALLIGHTFALLOFF
#define SPECULARAA
#define CLEARCOAT_TEXTUREDIRECTUV 0
#define CLEARCOAT_BUMPDIRECTUV 0
#define CLEARCOAT_TINT_TEXTUREDIRECTUV 0
#define ANISOTROPIC_TEXTUREDIRECTUV 0
#define BRDF_V_HEIGHT_CORRELATED
#define MS_BRDF_ENERGY_CONSERVATION
#define SHEEN_TEXTUREDIRECTUV 0
#define SS_THICKNESSANDMASK_TEXTUREDIRECTUV 0
#define DEBUGMODE 0

#define SHADER_NAME vertex:pbr
precision highp float;
layout(std140,column_major) uniform;
layout(set = 0, binding = 0) uniform Scene
{
mat4 viewProjection;
mat4 view;
};
layout(set = 1, binding = 0) uniform Material
{
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
layout(set = 1, binding = 1) uniform Mesh
{
mat4 world;
float visibility;
};
#define CUSTOM_VERTEX_BEGIN
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
const float PI=3.1415926535897932384626433832795;
const float LinearEncodePowerApprox=2.2;
const float GammaEncodePowerApprox=1.0/LinearEncodePowerApprox;
const vec3 LuminanceEncodeApprox=vec3(0.2126,0.7152,0.0722);
const float Epsilon=0.0000001;
#define saturate(x) clamp(x,0.0,1.0)
#define absEps(x) abs(x)+Epsilon
#define maxEps(x) max(x,Epsilon)
#define saturateEps(x) clamp(x,Epsilon,1.0)
mat3 transposeMat3(mat3 inMatrix) {
vec3 i0=inMatrix[0];
vec3 i1=inMatrix[1];
vec3 i2=inMatrix[2];
mat3 outMatrix=mat3(
vec3(i0.x,i1.x,i2.x),
vec3(i0.y,i1.y,i2.y),
vec3(i0.z,i1.z,i2.z)
);
return outMatrix;
}
mat3 inverseMat3(mat3 inMatrix) {
float a00=inMatrix[0][0],a01=inMatrix[0][1],a02=inMatrix[0][2];
float a10=inMatrix[1][0],a11=inMatrix[1][1],a12=inMatrix[1][2];
float a20=inMatrix[2][0],a21=inMatrix[2][1],a22=inMatrix[2][2];
float b01=a22*a11-a12*a21;
float b11=-a22*a10+a12*a20;
float b21=a21*a10-a11*a20;
float det=a00*b01+a01*b11+a02*b21;
return mat3(b01,(-a22*a01+a02*a21),(a12*a01-a02*a11),
b11,(a22*a00-a02*a20),(-a12*a00+a02*a10),
b21,(-a21*a00+a01*a20),(a11*a00-a01*a10))/det;
}
vec3 toLinearSpace(vec3 color)
{
return pow(color,vec3(LinearEncodePowerApprox));
}
vec3 toGammaSpace(vec3 color)
{
return pow(color,vec3(GammaEncodePowerApprox));
}
float square(float value)
{
return value*value;
}
float pow5(float value) {
float sq=value*value;
return sq*sq*value;
}
float getLuminance(vec3 color)
{
return clamp(dot(color,LuminanceEncodeApprox),0.,1.);
}
float getRand(vec2 seed) {
return fract(sin(dot(seed.xy ,vec2(12.9898,78.233)))*43758.5453);
}
float dither(vec2 seed,float varianceAmount) {
float rand=getRand(seed);
float dither=mix(-varianceAmount/255.0,varianceAmount/255.0,rand);
return dither;
}
const float rgbdMaxRange=255.0;
vec4 toRGBD(vec3 color) {
float maxRGB=maxEps(max(color.r,max(color.g,color.b)));
float D=max(rgbdMaxRange/maxRGB,1.);
D=clamp(floor(D)/255.0,0.,1.);
vec3 rgb=color.rgb*D;
rgb=toGammaSpace(rgb);
return vec4(rgb,D);
}
vec3 fromRGBD(vec4 rgbd) {
rgbd.rgb=toLinearSpace(rgbd.rgb);
return rgbd.rgb/rgbd.a;
}
layout(location = 0) out vec3 vPositionW;
layout(location = 1) out vec3 vNormalW;
layout(location = 2) out vec2 vMainUV1;
#define CUSTOM_VERTEX_DEFINITIONS
void main(void) {
#define CUSTOM_VERTEX_MAIN_BEGIN
vec3 positionUpdated=position;
vec3 normalUpdated=normal;
vec2 uvUpdated=uv;
#define CUSTOM_VERTEX_UPDATE_POSITION
#define CUSTOM_VERTEX_UPDATE_NORMAL
mat4 finalWorld=world;
gl_Position=viewProjection*finalWorld*vec4(positionUpdated,1.0);
vec4 worldPos=finalWorld*vec4(positionUpdated,1.0);
vPositionW=vec3(worldPos);
mat3 normalWorld=mat3(finalWorld);
vNormalW=normalize(normalWorld*normalUpdated);
vec2 uv2=vec2(0.,0.);
vMainUV1=uvUpdated;
#define CUSTOM_VERTEX_MAIN_END
gl_Position.y *= -1.; }
`;

const fragmentShaderWHLSL1 = `
#version 450
#define PBR
#define MAINUV1
#define UV1
#define ALBEDO
#define ALBEDODIRECTUV 1
#define AMBIENT
#define AMBIENTDIRECTUV 1
#define AMBIENTINGRAYSCALE
#define OPACITYDIRECTUV 0
#define ALPHATESTVALUE 0.4
#define SPECULAROVERALPHA
#define RADIANCEOVERALPHA
#define EMISSIVE
#define EMISSIVEDIRECTUV 1
#define REFLECTIVITY
#define REFLECTIVITYDIRECTUV 1
#define LODBASEDMICROSFURACE
#define MICROSURFACEMAPDIRECTUV 0
#define METALLICWORKFLOW
#define ROUGHNESSSTOREINMETALMAPGREEN
#define METALLNESSSTOREINMETALMAPBLUE
#define ENVIRONMENTBRDF
#define NORMAL
#define BUMP
#define BUMPDIRECTUV 1
#define NORMALXYSCALE
#define LIGHTMAPDIRECTUV 0
#define REFLECTION
#define REFLECTIONMAP_3D
#define REFLECTIONMAP_CUBIC
#define USESPHERICALFROMREFLECTIONMAP
#define SPHERICAL_HARMONICS
#define RGBDREFLECTION
#define RADIANCEOCCLUSION
#define HORIZONOCCLUSION
#define NUM_BONE_INFLUENCERS 0
#define BonesPerMesh 0
#define NUM_MORPH_INFLUENCERS 0
#define VIGNETTEBLENDMODEMULTIPLY
#define SAMPLER3DGREENDEPTH
#define SAMPLER3DBGRMAP
#define USEPHYSICALLIGHTFALLOFF
#define SPECULARAA
#define CLEARCOAT_TEXTUREDIRECTUV 0
#define CLEARCOAT_BUMPDIRECTUV 0
#define CLEARCOAT_TINT_TEXTUREDIRECTUV 0
#define ANISOTROPIC_TEXTUREDIRECTUV 0
#define BRDF_V_HEIGHT_CORRELATED
#define MS_BRDF_ENERGY_CONSERVATION
#define SHEEN_TEXTUREDIRECTUV 0
#define SS_THICKNESSANDMASK_TEXTUREDIRECTUV 0
#define DEBUGMODE 0

#define SHADER_NAME fragment:pbr


#define CUSTOM_FRAGMENT_BEGIN
precision highp float;
#define FROMLINEARSPACE
layout(std140,column_major) uniform;
layout(set = 0, binding = 0) uniform Scene
{
mat4 viewProjection;
mat4 view;
};
layout(set = 1, binding = 0) uniform Material
{
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
layout(set = 1, binding = 1) uniform Mesh
{
mat4 world;
float visibility;
};
layout(location = 0) in vec3 vPositionW;
layout(location = 1) in vec3 vNormalW;
layout(location = 2) in vec2 vMainUV1;
layout(set = 0, binding = 1) uniform texture2D environmentBrdfSamplerTexture;
                    layout(set = 0, binding = 2) uniform sampler environmentBrdfSamplerSampler;
                    #define environmentBrdfSampler sampler2D(environmentBrdfSamplerTexture, environmentBrdfSamplerSampler)
#define sampleReflection(s,c) texture(s,c)
layout(set = 2, binding = 0) uniform textureCube reflectionSamplerTexture;
                    layout(set = 2, binding = 1) uniform sampler reflectionSamplerSampler;
                    #define reflectionSampler samplerCube(reflectionSamplerTexture, reflectionSamplerSampler)
#define sampleReflectionLod(s,c,l) textureLod(s,c,l)
#define vAlbedoUV vMainUV1
layout(set = 2, binding = 2) uniform texture2D albedoSamplerTexture;
                    layout(set = 2, binding = 3) uniform sampler albedoSamplerSampler;
                    #define albedoSampler sampler2D(albedoSamplerTexture, albedoSamplerSampler)
#define vReflectivityUV vMainUV1
layout(set = 2, binding = 4) uniform texture2D reflectivitySamplerTexture;
                    layout(set = 2, binding = 5) uniform sampler reflectivitySamplerSampler;
                    #define reflectivitySampler sampler2D(reflectivitySamplerTexture, reflectivitySamplerSampler)
#define vAmbientUV vMainUV1
layout(set = 2, binding = 6) uniform texture2D ambientSamplerTexture;
                    layout(set = 2, binding = 7) uniform sampler ambientSamplerSampler;
                    #define ambientSampler sampler2D(ambientSamplerTexture, ambientSamplerSampler)
#define vEmissiveUV vMainUV1
layout(set = 2, binding = 8) uniform texture2D emissiveSamplerTexture;
                    layout(set = 2, binding = 9) uniform sampler emissiveSamplerSampler;
                    #define emissiveSampler sampler2D(emissiveSamplerTexture, emissiveSamplerSampler)
#define vBumpUV vMainUV1
layout(set = 2, binding = 10) uniform texture2D bumpSamplerTexture;
                    layout(set = 2, binding = 11) uniform sampler bumpSamplerSampler;
                    #define bumpSampler sampler2D(bumpSamplerTexture, bumpSamplerSampler)
const float PI=3.1415926535897932384626433832795;
const float LinearEncodePowerApprox=2.2;
const float GammaEncodePowerApprox=1.0/LinearEncodePowerApprox;
const vec3 LuminanceEncodeApprox=vec3(0.2126,0.7152,0.0722);
const float Epsilon=0.0000001;
#define saturate(x) clamp(x,0.0,1.0)
#define absEps(x) abs(x)+Epsilon
#define maxEps(x) max(x,Epsilon)
#define saturateEps(x) clamp(x,Epsilon,1.0)
mat3 transposeMat3(mat3 inMatrix) {
vec3 i0=inMatrix[0];
vec3 i1=inMatrix[1];
vec3 i2=inMatrix[2];
mat3 outMatrix=mat3(
vec3(i0.x,i1.x,i2.x),
vec3(i0.y,i1.y,i2.y),
vec3(i0.z,i1.z,i2.z)
);
return outMatrix;
}
mat3 inverseMat3(mat3 inMatrix) {
float a00=inMatrix[0][0],a01=inMatrix[0][1],a02=inMatrix[0][2];
float a10=inMatrix[1][0],a11=inMatrix[1][1],a12=inMatrix[1][2];
float a20=inMatrix[2][0],a21=inMatrix[2][1],a22=inMatrix[2][2];
float b01=a22*a11-a12*a21;
float b11=-a22*a10+a12*a20;
float b21=a21*a10-a11*a20;
float det=a00*b01+a01*b11+a02*b21;
return mat3(b01,(-a22*a01+a02*a21),(a12*a01-a02*a11),
b11,(a22*a00-a02*a20),(-a12*a00+a02*a10),
b21,(-a21*a00+a01*a20),(a11*a00-a01*a10))/det;
}
vec3 toLinearSpace(vec3 color)
{
return pow(color,vec3(LinearEncodePowerApprox));
}
vec3 toGammaSpace(vec3 color)
{
return pow(color,vec3(GammaEncodePowerApprox));
}
float square(float value)
{
return value*value;
}
float pow5(float value) {
float sq=value*value;
return sq*sq*value;
}
float getLuminance(vec3 color)
{
return clamp(dot(color,LuminanceEncodeApprox),0.,1.);
}
float getRand(vec2 seed) {
return fract(sin(dot(seed.xy ,vec2(12.9898,78.233)))*43758.5453);
}
float dither(vec2 seed,float varianceAmount) {
float rand=getRand(seed);
float dither=mix(-varianceAmount/255.0,varianceAmount/255.0,rand);
return dither;
}
const float rgbdMaxRange=255.0;
vec4 toRGBD(vec3 color) {
float maxRGB=maxEps(max(color.r,max(color.g,color.b)));
float D=max(rgbdMaxRange/maxRGB,1.);
D=clamp(floor(D)/255.0,0.,1.);
vec3 rgb=color.rgb*D;
rgb=toGammaSpace(rgb);
return vec4(rgb,D);
}
vec3 fromRGBD(vec4 rgbd) {
rgbd.rgb=toLinearSpace(rgbd.rgb);
return rgbd.rgb/rgbd.a;
}
#define RECIPROCAL_PI2 0.15915494
#define RECIPROCAL_PI 0.31830988618
#define MINIMUMVARIANCE 0.0005
float convertRoughnessToAverageSlope(float roughness)
{
return square(roughness)+MINIMUMVARIANCE;
}
float fresnelGrazingReflectance(float reflectance0) {
float reflectance90=saturate(reflectance0*25.0);
return reflectance90;
}
vec2 getAARoughnessFactors(vec3 normalVector) {
vec3 nDfdx=dFdx(normalVector.xyz);
vec3 nDfdy=dFdy(normalVector.xyz);
float slopeSquare=max(dot(nDfdx,nDfdx),dot(nDfdy,nDfdy));
float geometricRoughnessFactor=pow(saturate(slopeSquare),0.333);
float geometricAlphaGFactor=sqrt(slopeSquare);
geometricAlphaGFactor*=0.75;
return vec2(geometricRoughnessFactor,geometricAlphaGFactor);
}
vec4 applyImageProcessing(vec4 result) {
result.rgb=toGammaSpace(result.rgb);
result.rgb=saturate(result.rgb);
return result;
}
vec3 computeEnvironmentIrradiance(vec3 normal) {
return vSphericalL00
+vSphericalL1_1*(normal.y)
+vSphericalL10*(normal.z)
+vSphericalL11*(normal.x)
+vSphericalL2_2*(normal.y*normal.x)
+vSphericalL2_1*(normal.y*normal.z)
+vSphericalL20*((3.0*normal.z*normal.z)-1.0)
+vSphericalL21*(normal.z*normal.x)
+vSphericalL22*(normal.x*normal.x-(normal.y*normal.y));
}
struct preLightingInfo
{
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
preLightingInfo computePointAndSpotPreLightingInfo(vec4 lightData,vec3 V,vec3 N) {
preLightingInfo result;
result.lightOffset=lightData.xyz-vPositionW;
result.lightDistanceSquared=dot(result.lightOffset,result.lightOffset);
result.lightDistance=sqrt(result.lightDistanceSquared);
result.L=normalize(result.lightOffset);
result.H=normalize(V+result.L);
result.VdotH=saturate(dot(V,result.H));
result.NdotLUnclamped=dot(N,result.L);
result.NdotL=saturateEps(result.NdotLUnclamped);
return result;
}
preLightingInfo computeDirectionalPreLightingInfo(vec4 lightData,vec3 V,vec3 N) {
preLightingInfo result;
result.lightDistance=length(-lightData.xyz);
result.L=normalize(-lightData.xyz);
result.H=normalize(V+result.L);
result.VdotH=saturate(dot(V,result.H));
result.NdotLUnclamped=dot(N,result.L);
result.NdotL=saturateEps(result.NdotLUnclamped);
return result;
}
preLightingInfo computeHemisphericPreLightingInfo(vec4 lightData,vec3 V,vec3 N) {
preLightingInfo result;
result.NdotL=dot(N,lightData.xyz)*0.5+0.5;
result.NdotL=saturateEps(result.NdotL);
result.NdotLUnclamped=result.NdotL;
return result;
}
float computeDistanceLightFalloff_Standard(vec3 lightOffset,float range)
{
return max(0.,1.0-length(lightOffset)/range);
}
float computeDistanceLightFalloff_Physical(float lightDistanceSquared)
{
return 1.0/maxEps(lightDistanceSquared);
}
float computeDistanceLightFalloff_GLTF(float lightDistanceSquared,float inverseSquaredRange)
{
float lightDistanceFalloff=1.0/maxEps(lightDistanceSquared);
float factor=lightDistanceSquared*inverseSquaredRange;
float attenuation=saturate(1.0-factor*factor);
attenuation*=attenuation;
lightDistanceFalloff*=attenuation;
return lightDistanceFalloff;
}
float computeDistanceLightFalloff(vec3 lightOffset,float lightDistanceSquared,float range,float inverseSquaredRange)
{
return computeDistanceLightFalloff_Physical(lightDistanceSquared);
}
float computeDirectionalLightFalloff_Standard(vec3 lightDirection,vec3 directionToLightCenterW,float cosHalfAngle,float exponent)
{
float falloff=0.0;
float cosAngle=maxEps(dot(-lightDirection,directionToLightCenterW));
if (cosAngle>=cosHalfAngle)
{
falloff=max(0.,pow(cosAngle,exponent));
}
return falloff;
}
float computeDirectionalLightFalloff_Physical(vec3 lightDirection,vec3 directionToLightCenterW,float cosHalfAngle)
{
const float kMinusLog2ConeAngleIntensityRatio=6.64385618977;
float concentrationKappa=kMinusLog2ConeAngleIntensityRatio/(1.0-cosHalfAngle);
vec4 lightDirectionSpreadSG=vec4(-lightDirection*concentrationKappa,-concentrationKappa);
float falloff=exp2(dot(vec4(directionToLightCenterW,1.0),lightDirectionSpreadSG));
return falloff;
}
float computeDirectionalLightFalloff_GLTF(vec3 lightDirection,vec3 directionToLightCenterW,float lightAngleScale,float lightAngleOffset)
{
float cd=dot(-lightDirection,directionToLightCenterW);
float falloff=saturate(cd*lightAngleScale+lightAngleOffset);
falloff*=falloff;
return falloff;
}
float computeDirectionalLightFalloff(vec3 lightDirection,vec3 directionToLightCenterW,float cosHalfAngle,float exponent,float lightAngleScale,float lightAngleOffset)
{
return computeDirectionalLightFalloff_Physical(lightDirection,directionToLightCenterW,cosHalfAngle);
}
#define FRESNEL_MAXIMUM_ON_ROUGH 0.25
vec3 getEnergyConservationFactor(const vec3 specularEnvironmentR0,const vec3 environmentBrdf) {
return 1.0+specularEnvironmentR0*(1.0/environmentBrdf.y-1.0);
}
vec3 getBRDFLookup(float NdotV,float perceptualRoughness) {
vec2 UV=vec2(NdotV,perceptualRoughness);
vec4 brdfLookup=texture(environmentBrdfSampler,UV);
return brdfLookup.rgb;
}
vec3 getReflectanceFromBRDFLookup(const vec3 specularEnvironmentR0,const vec3 environmentBrdf) {
vec3 reflectance=mix(environmentBrdf.xxx,environmentBrdf.yyy,specularEnvironmentR0);
return reflectance;
}
vec3 fresnelSchlickGGX(float VdotH,vec3 reflectance0,vec3 reflectance90)
{
return reflectance0+(reflectance90-reflectance0)*pow5(1.0-VdotH);
}
float fresnelSchlickGGX(float VdotH,float reflectance0,float reflectance90)
{
return reflectance0+(reflectance90-reflectance0)*pow5(1.0-VdotH);
}
float normalDistributionFunction_TrowbridgeReitzGGX(float NdotH,float alphaG)
{
float a2=square(alphaG);
float d=NdotH*NdotH*(a2-1.0)+1.0;
return a2/(PI*d*d);
}
float smithVisibility_GGXCorrelated(float NdotL,float NdotV,float alphaG) {
float a2=alphaG*alphaG;
float GGXV=NdotL*sqrt(NdotV*(NdotV-a2*NdotV)+a2);
float GGXL=NdotV*sqrt(NdotL*(NdotL-a2*NdotL)+a2);
return 0.5/(GGXV+GGXL);
}
float diffuseBRDF_Burley(float NdotL,float NdotV,float VdotH,float roughness) {
float diffuseFresnelNV=pow5(saturateEps(1.0-NdotL));
float diffuseFresnelNL=pow5(saturateEps(1.0-NdotV));
float diffuseFresnel90=0.5+2.0*VdotH*VdotH*roughness;
float fresnel =
(1.0+(diffuseFresnel90-1.0)*diffuseFresnelNL) *
(1.0+(diffuseFresnel90-1.0)*diffuseFresnelNV);
return fresnel/PI;
}
#define CLEARCOATREFLECTANCE90 1.0
struct lightingInfo
{
vec3 diffuse;
};
float adjustRoughnessFromLightProperties(float roughness,float lightRadius,float lightDistance) {
float lightRoughness=lightRadius/lightDistance;
float totalRoughness=saturate(lightRoughness+roughness);
return totalRoughness;
}
vec3 computeHemisphericDiffuseLighting(preLightingInfo info,vec3 lightColor,vec3 groundColor) {
return mix(groundColor,lightColor,info.NdotL);
}
vec3 computeDiffuseLighting(preLightingInfo info,vec3 lightColor) {
float diffuseTerm=diffuseBRDF_Burley(info.NdotL,info.NdotV,info.VdotH,info.roughness);
return diffuseTerm*info.attenuation*info.NdotL*lightColor;
}
vec2 computeProjectionTextureDiffuseLightingUV(mat4 textureProjectionMatrix){
vec4 strq=textureProjectionMatrix*vec4(vPositionW,1.0);
strq/=strq.w;
return strq.xy;
}
float getLodFromAlphaG(float cubeMapDimensionPixels,float microsurfaceAverageSlope) {
float microsurfaceAverageSlopeTexels=cubeMapDimensionPixels*microsurfaceAverageSlope;
float lod=log2(microsurfaceAverageSlopeTexels);
return lod;
}
float getLinearLodFromRoughness(float cubeMapDimensionPixels,float roughness) {
float lod=log2(cubeMapDimensionPixels)*roughness;
return lod;
}
float environmentRadianceOcclusion(float ambientOcclusion,float NdotVUnclamped) {
float temp=NdotVUnclamped+ambientOcclusion;
return saturate(square(temp)-1.0+ambientOcclusion);
}
float environmentHorizonOcclusion(vec3 view,vec3 normal) {
vec3 reflection=reflect(view,normal);
float temp=saturate(1.0+1.1*dot(reflection,normal));
return square(temp);
}
vec3 perturbNormal(mat3 cotangentFrame,vec3 textureSample,float scale)
{
textureSample=textureSample*2.0-1.0;
textureSample=normalize(textureSample*vec3(scale,scale,1.0));
return normalize(cotangentFrame*textureSample);
}
mat3 cotangent_frame(vec3 normal,vec3 p,vec2 uv,vec2 tangentSpaceParams)
{
uv=gl_FrontFacing ? uv : -uv;
vec3 dp1=dFdx(p);
vec3 dp2=dFdy(p);
vec2 duv1=dFdx(uv);
vec2 duv2=dFdy(uv);
vec3 dp2perp=cross(dp2,normal);
vec3 dp1perp=cross(normal,dp1);
vec3 tangent=dp2perp*duv1.x+dp1perp*duv2.x;
vec3 bitangent=dp2perp*duv1.y+dp1perp*duv2.y;
tangent*=tangentSpaceParams.x;
bitangent*=tangentSpaceParams.y;
float invmax=inversesqrt(max(dot(tangent,tangent),dot(bitangent,bitangent)));
return mat3(tangent*invmax,bitangent*invmax,normal);
}
vec3 perturbNormal(mat3 cotangentFrame,vec2 uv)
{
return perturbNormal(cotangentFrame,texture(bumpSampler,uv).xyz,vBumpInfos.y);
}
vec3 perturbNormal(mat3 cotangentFrame,vec3 color)
{
return perturbNormal(cotangentFrame,color,vBumpInfos.y);
}
mat3 cotangent_frame(vec3 normal,vec3 p,vec2 uv)
{
return cotangent_frame(normal,p,uv,vTangentSpaceParams);
}
vec3 parallaxCorrectNormal( vec3 vertexPos,vec3 origVec,vec3 cubeSize,vec3 cubePos ) {
vec3 invOrigVec=vec3(1.0,1.0,1.0)/origVec;
vec3 halfSize=cubeSize*0.5;
vec3 intersecAtMaxPlane=(cubePos+halfSize-vertexPos)*invOrigVec;
vec3 intersecAtMinPlane=(cubePos-halfSize-vertexPos)*invOrigVec;
vec3 largestIntersec=max(intersecAtMaxPlane,intersecAtMinPlane);
float distance=min(min(largestIntersec.x,largestIntersec.y),largestIntersec.z);
vec3 intersectPositionWS=vertexPos+origVec*distance;
return intersectPositionWS-cubePos;
}
vec3 computeFixedEquirectangularCoords(vec4 worldPos,vec3 worldNormal,vec3 direction)
{
float lon=atan(direction.z,direction.x);
float lat=acos(direction.y);
vec2 sphereCoords=vec2(lon,lat)*RECIPROCAL_PI2*2.0;
float s=sphereCoords.x*0.5+0.5;
float t=sphereCoords.y;
return vec3(s,t,0);
}
vec3 computeMirroredFixedEquirectangularCoords(vec4 worldPos,vec3 worldNormal,vec3 direction)
{
float lon=atan(direction.z,direction.x);
float lat=acos(direction.y);
vec2 sphereCoords=vec2(lon,lat)*RECIPROCAL_PI2*2.0;
float s=sphereCoords.x*0.5+0.5;
float t=sphereCoords.y;
return vec3(1.0-s,t,0);
}
vec3 computeEquirectangularCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix)
{
vec3 cameraToVertex=normalize(worldPos.xyz-eyePosition);
vec3 r=normalize(reflect(cameraToVertex,worldNormal));
r=vec3(reflectionMatrix*vec4(r,0));
float lon=atan(r.z,r.x);
float lat=acos(r.y);
vec2 sphereCoords=vec2(lon,lat)*RECIPROCAL_PI2*2.0;
float s=sphereCoords.x*0.5+0.5;
float t=sphereCoords.y;
return vec3(s,t,0);
}
vec3 computeSphericalCoords(vec4 worldPos,vec3 worldNormal,mat4 view,mat4 reflectionMatrix)
{
vec3 viewDir=normalize(vec3(view*worldPos));
vec3 viewNormal=normalize(vec3(view*vec4(worldNormal,0.0)));
vec3 r=reflect(viewDir,viewNormal);
r=vec3(reflectionMatrix*vec4(r,0));
r.z=r.z-1.0;
float m=2.0*length(r);
return vec3(r.x/m+0.5,1.0-r.y/m-0.5,0);
}
vec3 computePlanarCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix)
{
vec3 viewDir=worldPos.xyz-eyePosition;
vec3 coords=normalize(reflect(viewDir,worldNormal));
return vec3(reflectionMatrix*vec4(coords,1));
}
vec3 computeCubicCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix)
{
vec3 viewDir=normalize(worldPos.xyz-eyePosition);
vec3 coords=reflect(viewDir,worldNormal);
coords=vec3(reflectionMatrix*vec4(coords,0));
return coords;
}
vec3 computeCubicLocalCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix,vec3 reflectionSize,vec3 reflectionPosition)
{
vec3 viewDir=normalize(worldPos.xyz-eyePosition);
vec3 coords=reflect(viewDir,worldNormal);
coords=parallaxCorrectNormal(worldPos.xyz,coords,reflectionSize,reflectionPosition);
coords=vec3(reflectionMatrix*vec4(coords,0));
return coords;
}
vec3 computeProjectionCoords(vec4 worldPos,mat4 view,mat4 reflectionMatrix)
{
return vec3(reflectionMatrix*(view*worldPos));
}
vec3 computeSkyBoxCoords(vec3 positionW,mat4 reflectionMatrix)
{
return vec3(reflectionMatrix*vec4(positionW,0));
}
vec3 computeReflectionCoords(vec4 worldPos,vec3 worldNormal)
{
return computeCubicCoords(worldPos,worldNormal,vEyePosition.xyz,reflectionMatrix);
}
#define CUSTOM_FRAGMENT_DEFINITIONS
layout(location = 0) out vec4 glFragColor;
void main(void) {
#define CUSTOM_FRAGMENT_MAIN_BEGIN
vec3 viewDirectionW=normalize(vEyePosition.xyz-vPositionW);
vec3 normalW=normalize(vNormalW);
vec2 uvOffset=vec2(0.0,0.0);
float normalScale=1.0;
mat3 TBN=cotangent_frame(normalW*normalScale,vPositionW,vBumpUV);
normalW=perturbNormal(TBN,vBumpUV+uvOffset);
vec3 surfaceAlbedo=vAlbedoColor.rgb;
float alpha=vAlbedoColor.a;
vec4 albedoTexture=texture(albedoSampler,vAlbedoUV+uvOffset);
surfaceAlbedo*=toLinearSpace(albedoTexture.rgb);
surfaceAlbedo*=vAlbedoInfos.y;
#define CUSTOM_FRAGMENT_UPDATE_ALBEDO
#define CUSTOM_FRAGMENT_UPDATE_ALPHA
#define CUSTOM_FRAGMENT_BEFORE_LIGHTS
vec3 ambientOcclusionColor=vec3(1.,1.,1.);
vec3 ambientOcclusionColorMap=texture(ambientSampler,vAmbientUV+uvOffset).rgb*vAmbientInfos.y;
ambientOcclusionColorMap=vec3(ambientOcclusionColorMap.r,ambientOcclusionColorMap.r,ambientOcclusionColorMap.r);
ambientOcclusionColor=mix(ambientOcclusionColor,ambientOcclusionColorMap,vAmbientInfos.z);
float microSurface=vReflectivityColor.a;
vec3 surfaceReflectivityColor=vReflectivityColor.rgb;
vec2 metallicRoughness=surfaceReflectivityColor.rg;
vec4 surfaceMetallicColorMap=texture(reflectivitySampler,vReflectivityUV+uvOffset);
metallicRoughness.r*=surfaceMetallicColorMap.b;
metallicRoughness.g*=surfaceMetallicColorMap.g;
#define CUSTOM_FRAGMENT_UPDATE_METALLICROUGHNESS
microSurface=1.0-metallicRoughness.g;
vec3 baseColor=surfaceAlbedo;
const vec3 DefaultSpecularReflectanceDielectric=vec3(0.04,0.04,0.04);
surfaceAlbedo=mix(baseColor.rgb*(1.0-DefaultSpecularReflectanceDielectric.r),vec3(0.,0.,0.),metallicRoughness.r);
surfaceReflectivityColor=mix(DefaultSpecularReflectanceDielectric,baseColor,metallicRoughness.r);
microSurface=saturate(microSurface);
float roughness=1.-microSurface;
float NdotVUnclamped=dot(normalW,viewDirectionW);
float NdotV=absEps(NdotVUnclamped);
float alphaG=convertRoughnessToAverageSlope(roughness);
vec2 AARoughnessFactors=getAARoughnessFactors(normalW.xyz);
alphaG+=AARoughnessFactors.y;
vec4 environmentRadiance=vec4(0.,0.,0.,0.);
vec3 environmentIrradiance=vec3(0.,0.,0.);
vec3 reflectionVector=computeReflectionCoords(vec4(vPositionW,1.0),normalW);
vec3 reflectionCoords=reflectionVector;
float reflectionLOD=getLodFromAlphaG(vReflectionMicrosurfaceInfos.x,alphaG);
reflectionLOD=reflectionLOD*vReflectionMicrosurfaceInfos.y+vReflectionMicrosurfaceInfos.z;
float requestedReflectionLOD=reflectionLOD;
environmentRadiance=sampleReflectionLod(reflectionSampler,reflectionCoords,requestedReflectionLOD);
environmentRadiance.rgb=fromRGBD(environmentRadiance);
vec3 irradianceVector=vec3(reflectionMatrix*vec4(normalW,0)).xyz;
environmentIrradiance=computeEnvironmentIrradiance(irradianceVector);
environmentRadiance.rgb*=vReflectionInfos.x;
environmentRadiance.rgb*=vReflectionColor.rgb;
environmentIrradiance*=vReflectionColor.rgb;
float reflectance=max(max(surfaceReflectivityColor.r,surfaceReflectivityColor.g),surfaceReflectivityColor.b);
float reflectance90=fresnelGrazingReflectance(reflectance);
vec3 specularEnvironmentR0=surfaceReflectivityColor.rgb;
vec3 specularEnvironmentR90=vec3(1.0,1.0,1.0)*reflectance90;
vec3 environmentBrdf=getBRDFLookup(NdotV,roughness);
vec3 energyConservationFactor=getEnergyConservationFactor(specularEnvironmentR0,environmentBrdf);
vec3 diffuseBase=vec3(0.,0.,0.);
preLightingInfo preInfo;
lightingInfo info;
float shadow=1.;
vec3 specularEnvironmentReflectance=getReflectanceFromBRDFLookup(specularEnvironmentR0,environmentBrdf);
float ambientMonochrome=ambientOcclusionColor.r;
float seo=environmentRadianceOcclusion(ambientMonochrome,NdotVUnclamped);
specularEnvironmentReflectance*=seo;
float eho=environmentHorizonOcclusion(-viewDirectionW,normalW);
specularEnvironmentReflectance*=eho;
vec3 finalIrradiance=environmentIrradiance;
finalIrradiance*=surfaceAlbedo.rgb;
vec3 finalRadiance=environmentRadiance.rgb;
finalRadiance*=specularEnvironmentReflectance;
vec3 finalRadianceScaled=finalRadiance*vLightingIntensity.z;
finalRadianceScaled*=energyConservationFactor;
vec3 finalDiffuse=diffuseBase;
finalDiffuse*=surfaceAlbedo.rgb;
finalDiffuse=max(finalDiffuse,0.0);
vec3 finalAmbient=vAmbientColor;
finalAmbient*=surfaceAlbedo.rgb;
vec3 finalEmissive=vEmissiveColor;
vec3 emissiveColorTex=texture(emissiveSampler,vEmissiveUV+uvOffset).rgb;
finalEmissive*=toLinearSpace(emissiveColorTex.rgb);
finalEmissive*=vEmissiveInfos.y;
vec3 ambientOcclusionForDirectDiffuse=mix(vec3(1.),ambientOcclusionColor,vAmbientInfos.w);
vec4 finalColor=vec4(
finalAmbient*ambientOcclusionColor +
finalDiffuse*ambientOcclusionForDirectDiffuse*vLightingIntensity.x +
finalIrradiance*ambientOcclusionColor*vLightingIntensity.z +
finalRadianceScaled +
finalEmissive*vLightingIntensity.y,
alpha);
#define CUSTOM_FRAGMENT_BEFORE_FOG
finalColor=max(finalColor,0.0);
finalColor=applyImageProcessing(finalColor);
finalColor.a*=visibility;
#define CUSTOM_FRAGMENT_BEFORE_FRAGCOLOR
glFragColor=finalColor;
}

`;

const vertexShaderWHLSL2 = `
#version 450
#define PBR
#define ALBEDODIRECTUV 0
#define AMBIENTDIRECTUV 0
#define OPACITYDIRECTUV 0
#define ALPHATESTVALUE 0.4
#define SPECULAROVERALPHA
#define RADIANCEOVERALPHA
#define EMISSIVEDIRECTUV 0
#define REFLECTIVITYDIRECTUV 0
#define LODBASEDMICROSFURACE
#define MICROSURFACEMAPDIRECTUV 0
#define ENVIRONMENTBRDF
#define NORMAL
#define BUMPDIRECTUV 0
#define NORMALXYSCALE
#define LIGHTMAPDIRECTUV 0
#define REFLECTION
#define REFLECTIONMAP_3D
#define REFLECTIONMAP_SKYBOX
#define SPHERICAL_HARMONICS
#define RGBDREFLECTION
#define RADIANCEOCCLUSION
#define HORIZONOCCLUSION
#define NUM_BONE_INFLUENCERS 0
#define BonesPerMesh 0
#define NUM_MORPH_INFLUENCERS 0
#define VIGNETTEBLENDMODEMULTIPLY
#define SAMPLER3DGREENDEPTH
#define SAMPLER3DBGRMAP
#define USEPHYSICALLIGHTFALLOFF
#define TWOSIDEDLIGHTING
#define CLEARCOAT_TEXTUREDIRECTUV 0
#define CLEARCOAT_BUMPDIRECTUV 0
#define CLEARCOAT_TINT_TEXTUREDIRECTUV 0
#define ANISOTROPIC_TEXTUREDIRECTUV 0
#define BRDF_V_HEIGHT_CORRELATED
#define MS_BRDF_ENERGY_CONSERVATION
#define SHEEN_TEXTUREDIRECTUV 0
#define SS_THICKNESSANDMASK_TEXTUREDIRECTUV 0
#define DEBUGMODE 0

#define SHADER_NAME vertex:pbr
precision highp float;
layout(std140,column_major) uniform;
layout(set = 0, binding = 0) uniform Scene
{
mat4 viewProjection;
mat4 view;
};
layout(set = 1, binding = 0) uniform Material
{
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
layout(set = 1, binding = 1) uniform Mesh
{
mat4 world;
float visibility;
};
#define CUSTOM_VERTEX_BEGIN
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
const float PI=3.1415926535897932384626433832795;
const float LinearEncodePowerApprox=2.2;
const float GammaEncodePowerApprox=1.0/LinearEncodePowerApprox;
const vec3 LuminanceEncodeApprox=vec3(0.2126,0.7152,0.0722);
const float Epsilon=0.0000001;
#define saturate(x) clamp(x,0.0,1.0)
#define absEps(x) abs(x)+Epsilon
#define maxEps(x) max(x,Epsilon)
#define saturateEps(x) clamp(x,Epsilon,1.0)
mat3 transposeMat3(mat3 inMatrix) {
vec3 i0=inMatrix[0];
vec3 i1=inMatrix[1];
vec3 i2=inMatrix[2];
mat3 outMatrix=mat3(
vec3(i0.x,i1.x,i2.x),
vec3(i0.y,i1.y,i2.y),
vec3(i0.z,i1.z,i2.z)
);
return outMatrix;
}
mat3 inverseMat3(mat3 inMatrix) {
float a00=inMatrix[0][0],a01=inMatrix[0][1],a02=inMatrix[0][2];
float a10=inMatrix[1][0],a11=inMatrix[1][1],a12=inMatrix[1][2];
float a20=inMatrix[2][0],a21=inMatrix[2][1],a22=inMatrix[2][2];
float b01=a22*a11-a12*a21;
float b11=-a22*a10+a12*a20;
float b21=a21*a10-a11*a20;
float det=a00*b01+a01*b11+a02*b21;
return mat3(b01,(-a22*a01+a02*a21),(a12*a01-a02*a11),
b11,(a22*a00-a02*a20),(-a12*a00+a02*a10),
b21,(-a21*a00+a01*a20),(a11*a00-a01*a10))/det;
}
vec3 toLinearSpace(vec3 color)
{
return pow(color,vec3(LinearEncodePowerApprox));
}
vec3 toGammaSpace(vec3 color)
{
return pow(color,vec3(GammaEncodePowerApprox));
}
float square(float value)
{
return value*value;
}
float pow5(float value) {
float sq=value*value;
return sq*sq*value;
}
float getLuminance(vec3 color)
{
return clamp(dot(color,LuminanceEncodeApprox),0.,1.);
}
float getRand(vec2 seed) {
return fract(sin(dot(seed.xy ,vec2(12.9898,78.233)))*43758.5453);
}
float dither(vec2 seed,float varianceAmount) {
float rand=getRand(seed);
float dither=mix(-varianceAmount/255.0,varianceAmount/255.0,rand);
return dither;
}
const float rgbdMaxRange=255.0;
vec4 toRGBD(vec3 color) {
float maxRGB=maxEps(max(color.r,max(color.g,color.b)));
float D=max(rgbdMaxRange/maxRGB,1.);
D=clamp(floor(D)/255.0,0.,1.);
vec3 rgb=color.rgb*D;
rgb=toGammaSpace(rgb);
return vec4(rgb,D);
}
vec3 fromRGBD(vec4 rgbd) {
rgbd.rgb=toLinearSpace(rgbd.rgb);
return rgbd.rgb/rgbd.a;
}
layout(location = 0) out vec3 vPositionW;
layout(location = 1) out vec3 vNormalW;
layout(location = 2) out vec3 vPositionUVW;
#define CUSTOM_VERTEX_DEFINITIONS
void main(void) {
#define CUSTOM_VERTEX_MAIN_BEGIN
vec3 positionUpdated=position;
vec3 normalUpdated=normal;
vPositionUVW=positionUpdated;
#define CUSTOM_VERTEX_UPDATE_POSITION
#define CUSTOM_VERTEX_UPDATE_NORMAL
mat4 finalWorld=world;
gl_Position=viewProjection*finalWorld*vec4(positionUpdated,1.0);
vec4 worldPos=finalWorld*vec4(positionUpdated,1.0);
vPositionW=vec3(worldPos);
mat3 normalWorld=mat3(finalWorld);
vNormalW=normalize(normalWorld*normalUpdated);
vec2 uvUpdated=vec2(0.,0.);
vec2 uv2=vec2(0.,0.);
#define CUSTOM_VERTEX_MAIN_END
gl_Position.y *= -1.; }
`;

const fragmentShaderWHLSL2 = `
#version 450
#define PBR
#define ALBEDODIRECTUV 0
#define AMBIENTDIRECTUV 0
#define OPACITYDIRECTUV 0
#define ALPHATESTVALUE 0.4
#define SPECULAROVERALPHA
#define RADIANCEOVERALPHA
#define EMISSIVEDIRECTUV 0
#define REFLECTIVITYDIRECTUV 0
#define LODBASEDMICROSFURACE
#define MICROSURFACEMAPDIRECTUV 0
#define ENVIRONMENTBRDF
#define NORMAL
#define BUMPDIRECTUV 0
#define NORMALXYSCALE
#define LIGHTMAPDIRECTUV 0
#define REFLECTION
#define REFLECTIONMAP_3D
#define REFLECTIONMAP_SKYBOX
#define SPHERICAL_HARMONICS
#define RGBDREFLECTION
#define RADIANCEOCCLUSION
#define HORIZONOCCLUSION
#define NUM_BONE_INFLUENCERS 0
#define BonesPerMesh 0
#define NUM_MORPH_INFLUENCERS 0
#define VIGNETTEBLENDMODEMULTIPLY
#define SAMPLER3DGREENDEPTH
#define SAMPLER3DBGRMAP
#define USEPHYSICALLIGHTFALLOFF
#define TWOSIDEDLIGHTING
#define CLEARCOAT_TEXTUREDIRECTUV 0
#define CLEARCOAT_BUMPDIRECTUV 0
#define CLEARCOAT_TINT_TEXTUREDIRECTUV 0
#define ANISOTROPIC_TEXTUREDIRECTUV 0
#define BRDF_V_HEIGHT_CORRELATED
#define MS_BRDF_ENERGY_CONSERVATION
#define SHEEN_TEXTUREDIRECTUV 0
#define SS_THICKNESSANDMASK_TEXTUREDIRECTUV 0
#define DEBUGMODE 0

#define SHADER_NAME fragment:pbr

#define CUSTOM_FRAGMENT_BEGIN
precision highp float;
#define FROMLINEARSPACE
layout(std140,column_major) uniform;
layout(set = 0, binding = 0) uniform Scene
{
mat4 viewProjection;
mat4 view;
};
layout(set = 1, binding = 0) uniform Material
{
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
layout(set = 1, binding = 1) uniform Mesh
{
mat4 world;
float visibility;
};
layout(location = 0) in vec3 vPositionW;
layout(location = 1) in vec3 vNormalW;
layout(set = 0, binding = 1) uniform texture2D environmentBrdfSamplerTexture;
                    layout(set = 0, binding = 2) uniform sampler environmentBrdfSamplerSampler;
                    #define environmentBrdfSampler sampler2D(environmentBrdfSamplerTexture, environmentBrdfSamplerSampler)
#define sampleReflection(s,c) texture(s,c)
layout(set = 2, binding = 0) uniform textureCube reflectionSamplerTexture;
                    layout(set = 2, binding = 1) uniform sampler reflectionSamplerSampler;
                    #define reflectionSampler samplerCube(reflectionSamplerTexture, reflectionSamplerSampler)
#define sampleReflectionLod(s,c,l) textureLod(s,c,l)
layout(location = 2) in vec3 vPositionUVW;
const float PI=3.1415926535897932384626433832795;
const float LinearEncodePowerApprox=2.2;
const float GammaEncodePowerApprox=1.0/LinearEncodePowerApprox;
const vec3 LuminanceEncodeApprox=vec3(0.2126,0.7152,0.0722);
const float Epsilon=0.0000001;
#define saturate(x) clamp(x,0.0,1.0)
#define absEps(x) abs(x)+Epsilon
#define maxEps(x) max(x,Epsilon)
#define saturateEps(x) clamp(x,Epsilon,1.0)
mat3 transposeMat3(mat3 inMatrix) {
vec3 i0=inMatrix[0];
vec3 i1=inMatrix[1];
vec3 i2=inMatrix[2];
mat3 outMatrix=mat3(
vec3(i0.x,i1.x,i2.x),
vec3(i0.y,i1.y,i2.y),
vec3(i0.z,i1.z,i2.z)
);
return outMatrix;
}
mat3 inverseMat3(mat3 inMatrix) {
float a00=inMatrix[0][0],a01=inMatrix[0][1],a02=inMatrix[0][2];
float a10=inMatrix[1][0],a11=inMatrix[1][1],a12=inMatrix[1][2];
float a20=inMatrix[2][0],a21=inMatrix[2][1],a22=inMatrix[2][2];
float b01=a22*a11-a12*a21;
float b11=-a22*a10+a12*a20;
float b21=a21*a10-a11*a20;
float det=a00*b01+a01*b11+a02*b21;
return mat3(b01,(-a22*a01+a02*a21),(a12*a01-a02*a11),
b11,(a22*a00-a02*a20),(-a12*a00+a02*a10),
b21,(-a21*a00+a01*a20),(a11*a00-a01*a10))/det;
}
vec3 toLinearSpace(vec3 color)
{
return pow(color,vec3(LinearEncodePowerApprox));
}
vec3 toGammaSpace(vec3 color)
{
return pow(color,vec3(GammaEncodePowerApprox));
}
float square(float value)
{
return value*value;
}
float pow5(float value) {
float sq=value*value;
return sq*sq*value;
}
float getLuminance(vec3 color)
{
return clamp(dot(color,LuminanceEncodeApprox),0.,1.);
}
float getRand(vec2 seed) {
return fract(sin(dot(seed.xy ,vec2(12.9898,78.233)))*43758.5453);
}
float dither(vec2 seed,float varianceAmount) {
float rand=getRand(seed);
float dither=mix(-varianceAmount/255.0,varianceAmount/255.0,rand);
return dither;
}
const float rgbdMaxRange=255.0;
vec4 toRGBD(vec3 color) {
float maxRGB=maxEps(max(color.r,max(color.g,color.b)));
float D=max(rgbdMaxRange/maxRGB,1.);
D=clamp(floor(D)/255.0,0.,1.);
vec3 rgb=color.rgb*D;
rgb=toGammaSpace(rgb);
return vec4(rgb,D);
}
vec3 fromRGBD(vec4 rgbd) {
rgbd.rgb=toLinearSpace(rgbd.rgb);
return rgbd.rgb/rgbd.a;
}
#define RECIPROCAL_PI2 0.15915494
#define RECIPROCAL_PI 0.31830988618
#define MINIMUMVARIANCE 0.0005
float convertRoughnessToAverageSlope(float roughness)
{
return square(roughness)+MINIMUMVARIANCE;
}
float fresnelGrazingReflectance(float reflectance0) {
float reflectance90=saturate(reflectance0*25.0);
return reflectance90;
}
vec2 getAARoughnessFactors(vec3 normalVector) {
return vec2(0.);
}
vec4 applyImageProcessing(vec4 result) {
result.rgb=toGammaSpace(result.rgb);
result.rgb=saturate(result.rgb);
return result;
}
struct preLightingInfo
{
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
preLightingInfo computePointAndSpotPreLightingInfo(vec4 lightData,vec3 V,vec3 N) {
preLightingInfo result;
result.lightOffset=lightData.xyz-vPositionW;
result.lightDistanceSquared=dot(result.lightOffset,result.lightOffset);
result.lightDistance=sqrt(result.lightDistanceSquared);
result.L=normalize(result.lightOffset);
result.H=normalize(V+result.L);
result.VdotH=saturate(dot(V,result.H));
result.NdotLUnclamped=dot(N,result.L);
result.NdotL=saturateEps(result.NdotLUnclamped);
return result;
}
preLightingInfo computeDirectionalPreLightingInfo(vec4 lightData,vec3 V,vec3 N) {
preLightingInfo result;
result.lightDistance=length(-lightData.xyz);
result.L=normalize(-lightData.xyz);
result.H=normalize(V+result.L);
result.VdotH=saturate(dot(V,result.H));
result.NdotLUnclamped=dot(N,result.L);
result.NdotL=saturateEps(result.NdotLUnclamped);
return result;
}
preLightingInfo computeHemisphericPreLightingInfo(vec4 lightData,vec3 V,vec3 N) {
preLightingInfo result;
result.NdotL=dot(N,lightData.xyz)*0.5+0.5;
result.NdotL=saturateEps(result.NdotL);
result.NdotLUnclamped=result.NdotL;
return result;
}
float computeDistanceLightFalloff_Standard(vec3 lightOffset,float range)
{
return max(0.,1.0-length(lightOffset)/range);
}
float computeDistanceLightFalloff_Physical(float lightDistanceSquared)
{
return 1.0/maxEps(lightDistanceSquared);
}
float computeDistanceLightFalloff_GLTF(float lightDistanceSquared,float inverseSquaredRange)
{
float lightDistanceFalloff=1.0/maxEps(lightDistanceSquared);
float factor=lightDistanceSquared*inverseSquaredRange;
float attenuation=saturate(1.0-factor*factor);
attenuation*=attenuation;
lightDistanceFalloff*=attenuation;
return lightDistanceFalloff;
}
float computeDistanceLightFalloff(vec3 lightOffset,float lightDistanceSquared,float range,float inverseSquaredRange)
{
return computeDistanceLightFalloff_Physical(lightDistanceSquared);
}
float computeDirectionalLightFalloff_Standard(vec3 lightDirection,vec3 directionToLightCenterW,float cosHalfAngle,float exponent)
{
float falloff=0.0;
float cosAngle=maxEps(dot(-lightDirection,directionToLightCenterW));
if (cosAngle>=cosHalfAngle)
{
falloff=max(0.,pow(cosAngle,exponent));
}
return falloff;
}
float computeDirectionalLightFalloff_Physical(vec3 lightDirection,vec3 directionToLightCenterW,float cosHalfAngle)
{
const float kMinusLog2ConeAngleIntensityRatio=6.64385618977;
float concentrationKappa=kMinusLog2ConeAngleIntensityRatio/(1.0-cosHalfAngle);
vec4 lightDirectionSpreadSG=vec4(-lightDirection*concentrationKappa,-concentrationKappa);
float falloff=exp2(dot(vec4(directionToLightCenterW,1.0),lightDirectionSpreadSG));
return falloff;
}
float computeDirectionalLightFalloff_GLTF(vec3 lightDirection,vec3 directionToLightCenterW,float lightAngleScale,float lightAngleOffset)
{
float cd=dot(-lightDirection,directionToLightCenterW);
float falloff=saturate(cd*lightAngleScale+lightAngleOffset);
falloff*=falloff;
return falloff;
}
float computeDirectionalLightFalloff(vec3 lightDirection,vec3 directionToLightCenterW,float cosHalfAngle,float exponent,float lightAngleScale,float lightAngleOffset)
{
return computeDirectionalLightFalloff_Physical(lightDirection,directionToLightCenterW,cosHalfAngle);
}
#define FRESNEL_MAXIMUM_ON_ROUGH 0.25
vec3 getEnergyConservationFactor(const vec3 specularEnvironmentR0,const vec3 environmentBrdf) {
return 1.0+specularEnvironmentR0*(1.0/environmentBrdf.y-1.0);
}
vec3 getBRDFLookup(float NdotV,float perceptualRoughness) {
vec2 UV=vec2(NdotV,perceptualRoughness);
vec4 brdfLookup=texture(environmentBrdfSampler,UV);
return brdfLookup.rgb;
}
vec3 getReflectanceFromBRDFLookup(const vec3 specularEnvironmentR0,const vec3 environmentBrdf) {
vec3 reflectance=mix(environmentBrdf.xxx,environmentBrdf.yyy,specularEnvironmentR0);
return reflectance;
}
vec3 getReflectanceFromAnalyticalBRDFLookup_Jones(float VdotN,vec3 reflectance0,vec3 reflectance90,float smoothness)
{
float weight=mix(FRESNEL_MAXIMUM_ON_ROUGH,1.0,smoothness);
return reflectance0+weight*(reflectance90-reflectance0)*pow5(saturate(1.0-VdotN));
}
vec3 fresnelSchlickGGX(float VdotH,vec3 reflectance0,vec3 reflectance90)
{
return reflectance0+(reflectance90-reflectance0)*pow5(1.0-VdotH);
}
float fresnelSchlickGGX(float VdotH,float reflectance0,float reflectance90)
{
return reflectance0+(reflectance90-reflectance0)*pow5(1.0-VdotH);
}
float normalDistributionFunction_TrowbridgeReitzGGX(float NdotH,float alphaG)
{
float a2=square(alphaG);
float d=NdotH*NdotH*(a2-1.0)+1.0;
return a2/(PI*d*d);
}
float smithVisibility_GGXCorrelated(float NdotL,float NdotV,float alphaG) {
float a2=alphaG*alphaG;
float GGXV=NdotL*sqrt(NdotV*(NdotV-a2*NdotV)+a2);
float GGXL=NdotV*sqrt(NdotL*(NdotL-a2*NdotL)+a2);
return 0.5/(GGXV+GGXL);
}
float diffuseBRDF_Burley(float NdotL,float NdotV,float VdotH,float roughness) {
float diffuseFresnelNV=pow5(saturateEps(1.0-NdotL));
float diffuseFresnelNL=pow5(saturateEps(1.0-NdotV));
float diffuseFresnel90=0.5+2.0*VdotH*VdotH*roughness;
float fresnel =
(1.0+(diffuseFresnel90-1.0)*diffuseFresnelNL) *
(1.0+(diffuseFresnel90-1.0)*diffuseFresnelNV);
return fresnel/PI;
}
#define CLEARCOATREFLECTANCE90 1.0
struct lightingInfo
{
vec3 diffuse;
};
float adjustRoughnessFromLightProperties(float roughness,float lightRadius,float lightDistance) {
float lightRoughness=lightRadius/lightDistance;
float totalRoughness=saturate(lightRoughness+roughness);
return totalRoughness;
}
vec3 computeHemisphericDiffuseLighting(preLightingInfo info,vec3 lightColor,vec3 groundColor) {
return mix(groundColor,lightColor,info.NdotL);
}
vec3 computeDiffuseLighting(preLightingInfo info,vec3 lightColor) {
float diffuseTerm=diffuseBRDF_Burley(info.NdotL,info.NdotV,info.VdotH,info.roughness);
return diffuseTerm*info.attenuation*info.NdotL*lightColor;
}
vec2 computeProjectionTextureDiffuseLightingUV(mat4 textureProjectionMatrix){
vec4 strq=textureProjectionMatrix*vec4(vPositionW,1.0);
strq/=strq.w;
return strq.xy;
}
float getLodFromAlphaG(float cubeMapDimensionPixels,float microsurfaceAverageSlope) {
float microsurfaceAverageSlopeTexels=cubeMapDimensionPixels*microsurfaceAverageSlope;
float lod=log2(microsurfaceAverageSlopeTexels);
return lod;
}
float getLinearLodFromRoughness(float cubeMapDimensionPixels,float roughness) {
float lod=log2(cubeMapDimensionPixels)*roughness;
return lod;
}
float environmentRadianceOcclusion(float ambientOcclusion,float NdotVUnclamped) {
float temp=NdotVUnclamped+ambientOcclusion;
return saturate(square(temp)-1.0+ambientOcclusion);
}
float environmentHorizonOcclusion(vec3 view,vec3 normal) {
vec3 reflection=reflect(view,normal);
float temp=saturate(1.0+1.1*dot(reflection,normal));
return square(temp);
}
vec3 parallaxCorrectNormal( vec3 vertexPos,vec3 origVec,vec3 cubeSize,vec3 cubePos ) {
vec3 invOrigVec=vec3(1.0,1.0,1.0)/origVec;
vec3 halfSize=cubeSize*0.5;
vec3 intersecAtMaxPlane=(cubePos+halfSize-vertexPos)*invOrigVec;
vec3 intersecAtMinPlane=(cubePos-halfSize-vertexPos)*invOrigVec;
vec3 largestIntersec=max(intersecAtMaxPlane,intersecAtMinPlane);
float distance=min(min(largestIntersec.x,largestIntersec.y),largestIntersec.z);
vec3 intersectPositionWS=vertexPos+origVec*distance;
return intersectPositionWS-cubePos;
}
vec3 computeFixedEquirectangularCoords(vec4 worldPos,vec3 worldNormal,vec3 direction)
{
float lon=atan(direction.z,direction.x);
float lat=acos(direction.y);
vec2 sphereCoords=vec2(lon,lat)*RECIPROCAL_PI2*2.0;
float s=sphereCoords.x*0.5+0.5;
float t=sphereCoords.y;
return vec3(s,t,0);
}
vec3 computeMirroredFixedEquirectangularCoords(vec4 worldPos,vec3 worldNormal,vec3 direction)
{
float lon=atan(direction.z,direction.x);
float lat=acos(direction.y);
vec2 sphereCoords=vec2(lon,lat)*RECIPROCAL_PI2*2.0;
float s=sphereCoords.x*0.5+0.5;
float t=sphereCoords.y;
return vec3(1.0-s,t,0);
}
vec3 computeEquirectangularCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix)
{
vec3 cameraToVertex=normalize(worldPos.xyz-eyePosition);
vec3 r=normalize(reflect(cameraToVertex,worldNormal));
r=vec3(reflectionMatrix*vec4(r,0));
float lon=atan(r.z,r.x);
float lat=acos(r.y);
vec2 sphereCoords=vec2(lon,lat)*RECIPROCAL_PI2*2.0;
float s=sphereCoords.x*0.5+0.5;
float t=sphereCoords.y;
return vec3(s,t,0);
}
vec3 computeSphericalCoords(vec4 worldPos,vec3 worldNormal,mat4 view,mat4 reflectionMatrix)
{
vec3 viewDir=normalize(vec3(view*worldPos));
vec3 viewNormal=normalize(vec3(view*vec4(worldNormal,0.0)));
vec3 r=reflect(viewDir,viewNormal);
r=vec3(reflectionMatrix*vec4(r,0));
r.z=r.z-1.0;
float m=2.0*length(r);
return vec3(r.x/m+0.5,1.0-r.y/m-0.5,0);
}
vec3 computePlanarCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix)
{
vec3 viewDir=worldPos.xyz-eyePosition;
vec3 coords=normalize(reflect(viewDir,worldNormal));
return vec3(reflectionMatrix*vec4(coords,1));
}
vec3 computeCubicCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix)
{
vec3 viewDir=normalize(worldPos.xyz-eyePosition);
vec3 coords=reflect(viewDir,worldNormal);
coords=vec3(reflectionMatrix*vec4(coords,0));
return coords;
}
vec3 computeCubicLocalCoords(vec4 worldPos,vec3 worldNormal,vec3 eyePosition,mat4 reflectionMatrix,vec3 reflectionSize,vec3 reflectionPosition)
{
vec3 viewDir=normalize(worldPos.xyz-eyePosition);
vec3 coords=reflect(viewDir,worldNormal);
coords=parallaxCorrectNormal(worldPos.xyz,coords,reflectionSize,reflectionPosition);
coords=vec3(reflectionMatrix*vec4(coords,0));
return coords;
}
vec3 computeProjectionCoords(vec4 worldPos,mat4 view,mat4 reflectionMatrix)
{
return vec3(reflectionMatrix*(view*worldPos));
}
vec3 computeSkyBoxCoords(vec3 positionW,mat4 reflectionMatrix)
{
return vec3(reflectionMatrix*vec4(positionW,0));
}
vec3 computeReflectionCoords(vec4 worldPos,vec3 worldNormal)
{
return computeSkyBoxCoords(vPositionUVW,reflectionMatrix);
}
#define CUSTOM_FRAGMENT_DEFINITIONS
layout(location = 0) out vec4 glFragColor;
void main(void) {
#define CUSTOM_FRAGMENT_MAIN_BEGIN
vec3 viewDirectionW=normalize(vEyePosition.xyz-vPositionW);
vec3 normalW=normalize(vNormalW);
vec2 uvOffset=vec2(0.0,0.0);
normalW=gl_FrontFacing ? normalW : -normalW;
vec3 surfaceAlbedo=vAlbedoColor.rgb;
float alpha=vAlbedoColor.a;
#define CUSTOM_FRAGMENT_UPDATE_ALBEDO
#define CUSTOM_FRAGMENT_UPDATE_ALPHA
#define CUSTOM_FRAGMENT_BEFORE_LIGHTS
vec3 ambientOcclusionColor=vec3(1.,1.,1.);
float microSurface=vReflectivityColor.a;
vec3 surfaceReflectivityColor=vReflectivityColor.rgb;
microSurface=saturate(microSurface);
float roughness=1.-microSurface;
float NdotVUnclamped=dot(normalW,viewDirectionW);
float NdotV=absEps(NdotVUnclamped);
float alphaG=convertRoughnessToAverageSlope(roughness);
vec2 AARoughnessFactors=getAARoughnessFactors(normalW.xyz);
vec4 environmentRadiance=vec4(0.,0.,0.,0.);
vec3 environmentIrradiance=vec3(0.,0.,0.);
vec3 reflectionVector=computeReflectionCoords(vec4(vPositionW,1.0),normalW);
vec3 reflectionCoords=reflectionVector;
float reflectionLOD=getLodFromAlphaG(vReflectionMicrosurfaceInfos.x,alphaG);
reflectionLOD=reflectionLOD*vReflectionMicrosurfaceInfos.y+vReflectionMicrosurfaceInfos.z;
float requestedReflectionLOD=reflectionLOD;
environmentRadiance=sampleReflectionLod(reflectionSampler,reflectionCoords,requestedReflectionLOD);
environmentRadiance.rgb=fromRGBD(environmentRadiance);
environmentRadiance.rgb*=vReflectionInfos.x;
environmentRadiance.rgb*=vReflectionColor.rgb;
environmentIrradiance*=vReflectionColor.rgb;
float reflectance=max(max(surfaceReflectivityColor.r,surfaceReflectivityColor.g),surfaceReflectivityColor.b);
float reflectance90=fresnelGrazingReflectance(reflectance);
vec3 specularEnvironmentR0=surfaceReflectivityColor.rgb;
vec3 specularEnvironmentR90=vec3(1.0,1.0,1.0)*reflectance90;
vec3 environmentBrdf=getBRDFLookup(NdotV,roughness);
vec3 energyConservationFactor=getEnergyConservationFactor(specularEnvironmentR0,environmentBrdf);
vec3 diffuseBase=vec3(0.,0.,0.);
preLightingInfo preInfo;
lightingInfo info;
float shadow=1.;
vec3 specularEnvironmentReflectance=getReflectanceFromAnalyticalBRDFLookup_Jones(NdotV,specularEnvironmentR0,specularEnvironmentR90,sqrt(microSurface));
surfaceAlbedo.rgb=(1.-reflectance)*surfaceAlbedo.rgb;
vec3 finalIrradiance=environmentIrradiance;
finalIrradiance*=surfaceAlbedo.rgb;
vec3 finalRadiance=environmentRadiance.rgb;
finalRadiance*=specularEnvironmentReflectance;
vec3 finalRadianceScaled=finalRadiance*vLightingIntensity.z;
finalRadianceScaled*=energyConservationFactor;
vec3 finalDiffuse=diffuseBase;
finalDiffuse*=surfaceAlbedo.rgb;
finalDiffuse=max(finalDiffuse,0.0);
vec3 finalAmbient=vAmbientColor;
finalAmbient*=surfaceAlbedo.rgb;
vec3 finalEmissive=vEmissiveColor;
vec3 ambientOcclusionForDirectDiffuse=ambientOcclusionColor;
vec4 finalColor=vec4(
finalAmbient*ambientOcclusionColor +
finalDiffuse*ambientOcclusionForDirectDiffuse*vLightingIntensity.x +
finalIrradiance*ambientOcclusionColor*vLightingIntensity.z +
finalRadianceScaled +
finalEmissive*vLightingIntensity.y,
alpha);
#define CUSTOM_FRAGMENT_BEFORE_FOG
finalColor=max(finalColor,0.0);
finalColor=applyImageProcessing(finalColor);
finalColor.a*=visibility;
#define CUSTOM_FRAGMENT_BEFORE_FRAGCOLOR
glFragColor=finalColor;
}

`;
