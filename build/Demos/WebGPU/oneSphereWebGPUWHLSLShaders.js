const vertexShaderWHLSL1 = `
        #include <metal_texture>
        #include <metal_matrix>
        using namespace metal;
        struct VertexInput {
            float3 position [[attribute(0)]];
            float3 normal [[attribute(1)]];
        };
        struct Scene {
            float4x4 viewProjection;
            float4x4 view;
        };
        struct BindGroupA {
            constant Scene* scene [[id(0)]];
            texture2d<float> aTexture [[id(1)]];
            sampler aSampler [[id(2)]];
            uint2 sceneBufferLength [[id(3)]];
        };
        struct Mesh {
            float4x4 world;
            float visibility;
        };
        struct BindGroupB {
            constant float* aBuffer [[id(0)]];
            constant Mesh* mesh [[id(1)]];
            uint2 aBufferLength [[id(2)]];
            uint2 meshBufferLength [[id(3)]];
        };
        struct BindGroupC {
            texture2d<float> aTexture [[id(0)]];
            sampler aSampler [[id(1)]];
        };
        vertex float4 vertexFunction(VertexInput vertexInput [[stage_in]], device BindGroupA& bindGroupA [[buffer(0)]], device BindGroupB& bindGroupB [[buffer(1)]], device BindGroupC& bindGroupC [[buffer(2)]]) {
            float3 positionUpdated = vertexInput.position;
            float3 normalUpdated = vertexInput.normal;
            float4x4 finalWorld = (bindGroupB.mesh[0].world);
            return (bindGroupA.scene[0].viewProjection) * finalWorld * float4(positionUpdated, 1.0);
        }
`;

const fragmentShaderWHLSL1 = `
        #include <metal_texture>
        #include <metal_matrix>
        using namespace metal;
        fragment float4 fragmentFunction() {
            return float4(0, 1, 1, 1);
        }
`;

const vertexShaderWHLSL2 = `
        #include <metal_texture>
        #include <metal_matrix>
        using namespace metal;
        struct VertexInput {
            float3 position [[attribute(0)]];
            float3 normal [[attribute(1)]];
        };
        struct Scene {
            float4x4 viewProjection;
            float4x4 view;
        };
        struct BindGroupA {
            constant Scene* scene [[id(0)]];
            texture2d<float> aTexture [[id(1)]];
            sampler aSampler [[id(2)]];
            uint2 sceneBufferLength [[id(3)]];
        };
        struct Mesh {
            float4x4 world;
            float visibility;
        };
        struct BindGroupB {
            constant float* aBuffer [[id(0)]];
            constant Mesh* mesh [[id(1)]];
            uint2 aBufferLength [[id(2)]];
            uint2 meshBufferLength [[id(3)]];
        };
        struct BindGroupC {
            texture2d<float> aTexture [[id(0)]];
            sampler aSampler [[id(1)]];
        };
        vertex float4 vertexFunction(VertexInput vertexInput [[stage_in]], device BindGroupA& bindGroupA [[buffer(0)]], device BindGroupB& bindGroupB [[buffer(1)]], device BindGroupC& bindGroupC [[buffer(2)]]) {
            float3 positionUpdated = vertexInput.position;
            float3 normalUpdated = vertexInput.normal;
            float4x4 finalWorld = (bindGroupB.mesh[0].world);
            return (bindGroupA.scene[0].viewProjection) * finalWorld * float4(positionUpdated, 1.0);
        }
`;

const fragmentShaderWHLSL2 = `
        #include <metal_texture>
        #include <metal_matrix>
        using namespace metal;
        fragment float4 fragmentFunction() {
            return float4(0, 1, 0, 1);
        }
`;
