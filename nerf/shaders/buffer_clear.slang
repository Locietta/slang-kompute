// nerf/shaders/buffer_clear.slang

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> buffer;

[[vk::push_constant]]
float clear_value;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= buffer.getCount()) return;
    buffer[id.x] = clear_value;
}
