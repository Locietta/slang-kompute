

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