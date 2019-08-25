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

struct BindGroupB {
    constant float[] aBuffer : register(b0, space1);
    constant Mesh[] mesh : register(b1, space1);
}

struct BindGroupC {
    Texture2D<float> aTexture : register(t0, space2);
    sampler aSampler : register(s1, space2);
}

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1), BindGroupA bindGroupA, BindGroupB bindGroupB, BindGroupC bindGroupC) {
    Output output;

    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    float4x4 finalWorld = bindGroupB.mesh[0].world;
    output.position = mul(mul(bindGroupA.scene[0].viewProjection, finalWorld), float4(positionUpdated, 1.0));

    output.vPositionW = float3(0, 0, 0);
    output.vNormalW = float3(0, 0, 0);
    output.vEnvironmentIrradiance = float3(0, 0, 0);
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

struct BindGroupB {
    constant float[] aBuffer : register(b0, space1);
    constant Mesh[] mesh : register(b1, space1);
}

struct BindGroupC {
    Texture2D<float> aTexture : register(t0, space2);
    sampler aSampler : register(s1, space2);
}

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1), BindGroupA bindGroupA, BindGroupB bindGroupB, BindGroupC bindGroupC) {
    Output output;
    
    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    float4x4 finalWorld = bindGroupB.mesh[0].world;
    output.position = mul(mul(bindGroupA.scene[0].viewProjection, finalWorld), float4(positionUpdated, 1.0));

    output.vPositionW = float3(0, 0, 0);
    output.vNormalW = float3(0, 0, 0);
    output.vPositionUVW = float3(0, 0, 0);
    return output;
}
`;

const fragmentShaderWHLSL2 = `
fragment float4 main(float3 vPositionW : attribute(0), float3 vNormalW : attribute(1), float3 vPositionUVW : attribute(2)) : SV_Target 0 {
    return float4(0, 1, 0, 1);
}
`;
