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

struct Mesh {
    float4x4 world;
    float visibility;
}

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1), constant Scene[] scene : register(b0, space0), constant Mesh[] mesh : register(b1, space1)) {
    Output output;

    float3 positionUpdated = position;
    float3 normalUpdated = normal;
    float4x4 finalWorld = transpose(mesh[0].world);
    output.position = mul(mul(transpose(scene[0].viewProjection), finalWorld), float4(positionUpdated, 1.0));

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

vertex Output main(float3 position : attribute(0), float3 normal : attribute(1)) {
    Output output;
    output.vPositionW = float3(0, 0, 0);
    output.vNormalW = float3(0, 0, 0);
    output.vPositionUVW = float3(0, 0, 0);
    output.position = float4(0, 0, 0, 0);
    return output;
}
`;

const fragmentShaderWHLSL2 = `
fragment float4 main(float3 vPositionW : attribute(0), float3 vNormalW : attribute(1), float3 vPositionUVW : attribute(2)) : SV_Target 0 {
    return float4(0, 1, 0, 1);
}
`;
