

target("test_dataset")
    set_kind("binary")
    set_default(false)
    add_files("test_dataset.cpp", "../dataset.cpp")
    add_packages("fmt", "stb", "nlohmann_json", "glm")
    add_tests("default")
    add_tests("args", {runargs = {"--dataset", "data/nerf_synthetic/lego", "--half_res"}})
    add_tests("full_res", {runargs = {"--dataset", "data/nerf_synthetic/lego", "--full_res"}})

target("test_ray_generator")
    set_kind("binary")
    set_default(false)
    add_files("test_ray_generator.cpp", "../ray_generator.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/ray_generator.slang")
    add_tests("default")

target("test_ray_sampler")
    set_kind("binary")
    set_default(false)
    add_files("test_ray_sampler.cpp", "../ray_generator.cpp", "../ray_sampler.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/ray_generator.slang", "../shaders/ray_sampler.slang")
    add_tests("default")

target("test_positional_encoder")
    set_kind("binary")
    set_default(false)
    add_files("test_positional_encoder.cpp", "../positional_encoder.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/positional_encoder.slang")
    add_tests("default")

target("test_nerf_network")
    set_kind("binary")
    set_default(false)
    add_files("test_nerf_network.cpp", "../nerf_network.cpp", "../positional_encoder.cpp")
    add_packages("kompute", "fmt", "glm", "access_private")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/nerf_*.slang", "../shaders/positional_encoder.slang")
    add_tests("default")

target("test_volume_renderer")
    set_kind("binary")
    set_default(false)
    add_files("test_volume_renderer.cpp", "../volume_renderer.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/volume_render.slang")
    add_tests("default")

target("test_buffer_clear")
    set_kind("binary")
    set_default(false)
    add_files("test_buffer_clear.cpp", "../OpClear.cpp")
    add_packages("kompute", "fmt", "glm")
    add_tests("default")

target("test_loss")
    set_kind("binary")
    set_default(true)
    add_files("test_loss.cpp", "../loss.cpp", "../OpClear.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/mse_*.slang")
    add_tests("default")

target("test_parallel_reduction")
    set_kind("binary")
    set_default(false)
    add_files("test_parallel_reduction.cpp", "../OpClear.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/parallel_reduction.slang")
    add_tests("default")

target("test_optimizer")
    set_kind("binary")
    set_default(false)
    add_files("test_optimizer.cpp", "../optimizer.cpp", "../OpClear.cpp")
    add_packages("kompute", "fmt", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("../shaders/adam_update.slang")
    add_tests("default")