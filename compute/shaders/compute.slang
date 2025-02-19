[SpecializationConstant]
const float m = 0.0;

// clang-format off

[[vk::binding(0, 0)]] RWStructuredBuffer<float> xi;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> xj;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> y;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> win;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> wouti;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> woutj;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> bin;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> bout;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> lout;

// clang-format on

float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

float inference(float2 x, float2 w, float b) {
    // Compute the linear mapping function
    float z = dot(w, x) + b;
    // Calculate the y-hat with sigmoid
    float yHat = sigmoid(z);
    return yHat;
}

float calculateLoss(float yHat, float y) {
    return -(y * log(yHat) + (1.0f - y) * log(1.0f - yHat));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint idx = DTid.x;

    float2 wCurr = float2(win[0], win[1]);
    float bCurr = bin[0];

    float2 xCurr = float2(xi[idx], xj[idx]);
    float yCurr = y[idx];

    float yHat = inference(xCurr, wCurr, bCurr);

    float dZ = yHat - yCurr;
    float2 dW = (1.0f / m) * xCurr * dZ;
    float dB = (1.0f / m) * dZ;
    wouti[idx] = dW.x;
    woutj[idx] = dW.y;
    bout[idx] = dB;

    lout[idx] = calculateLoss(yHat, yCurr);
}
