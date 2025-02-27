// Specialization constants
struct PushConstants {
    uint numPoints;
    uint posEncDim;
    uint hiddenDim;
};
[[vk::push_constant]]
PushConstants constants;

// Input tensors
[[vk::binding(0, 0)]]
RWStructuredBuffer<float> pos;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> dir;

// Network parameters - Layer 1
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> weightsL1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> biasL1;

// Network parameters - Layer 2
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> weightsL2;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> biasL2;

// Network parameters - Output layers
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> weightsRGB;
[[vk::binding(7, 0)]]
RWStructuredBuffer<float> biasRGB;
[[vk::binding(8, 0)]]
RWStructuredBuffer<float> weightsSigma;
[[vk::binding(9, 0)]]
RWStructuredBuffer<float> biasSigma;

// Output tensors
[[vk::binding(10, 0)]]
RWStructuredBuffer<float> rgb;
[[vk::binding(11, 0)]]
RWStructuredBuffer<float> sigma;

// Positional encoding function
void positionalEncoding(float x, uint offset, inout float encoded[126]) {
    for (uint i = 0; i < constants.posEncDim; i++) {
        encoded[offset + 2 * i] = sin(pow(2.0f, float(i)) * x);
        encoded[offset + 2 * i + 1] = cos(pow(2.0f, float(i)) * x);
    }
}

// ReLU activation function
float relu(float x) {
    return max(0.0f, x);
}

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint gID = DTid.x;
    if (gID >= constants.numPoints) return;

    // Get input point and view direction
    float x = pos[gID * 3];
    float y = pos[gID * 3 + 1];
    float z = pos[gID * 3 + 2];

    float dx = dir[gID * 3];
    float dy = dir[gID * 3 + 1];
    float dz = dir[gID * 3 + 2];

    // Apply positional encoding
    uint encDim = constants.posEncDim * 2;
    uint posEncSize = 3 * encDim;
    uint dirEncSize = 3 * encDim;
    uint totalEncSize = posEncSize + dirEncSize;

    float encoded[126];// 3 * 2 * posEncDim (position) + 3 * 2 * posEncDim (direction)

    // Position encoding
    positionalEncoding(x, 0, encoded);
    positionalEncoding(y, encDim, encoded);
    positionalEncoding(z, 2 * encDim, encoded);

    // Direction encoding
    positionalEncoding(dx, 3 * encDim, encoded);
    positionalEncoding(dy, 4 * encDim, encoded);
    positionalEncoding(dz, 5 * encDim, encoded);

    // Forward pass through MLP

    // First layer (using position encoding only)
    float hidden1[256];// hiddenDim
    for (uint i = 0; i < constants.hiddenDim; i++) {
        hidden1[i] = biasL1[i];
        for (uint j = 0; j < posEncSize; j++) {
            hidden1[i] += encoded[j] * weightsL1[i * posEncSize + j];
        }
        // ReLU activation
        hidden1[i] = relu(hidden1[i]);
    }

    // Second layer
    float hidden2[256];// hiddenDim
    for (uint i = 0; i < constants.hiddenDim; i++) {
        hidden2[i] = biasL2[i];
        for (uint j = 0; j < constants.hiddenDim; j++) {
            hidden2[i] += hidden1[j] * weightsL2[i * constants.hiddenDim + j];
        }
        // ReLU activation
        hidden2[i] = relu(hidden2[i]);
    }

    // Output RGB (using direction encoding)
    for (uint i = 0; i < 3; i++) {
        rgb[gID * 3 + i] = biasRGB[i];
        for (uint j = 0; j < constants.hiddenDim; j++) {
            rgb[gID * 3 + i] += hidden2[j] * weightsRGB[i * constants.hiddenDim + j];
        }
        // Additional processing for RGB values from direction
        for (uint j = 0; j < dirEncSize; j++) {
            rgb[gID * 3 + i] += encoded[posEncSize + j] * weightsRGB[(i * constants.hiddenDim) + constants.hiddenDim + j];
        }
        // Sigmoid activation for RGB
        rgb[gID * 3 + i] = sigmoid(rgb[gID * 3 + i]);
    }

    // Output sigma (density)
    sigma[gID] = biasSigma[0];
    for (uint j = 0; j < constants.hiddenDim; j++) {
        sigma[gID] += hidden2[j] * weightsSigma[j];
    }
    // ReLU for sigma to ensure positive density
    sigma[gID] = relu(sigma[gID]);
}
