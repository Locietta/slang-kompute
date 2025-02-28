

target("test_dataset")
    set_kind("binary")
    set_default(false)
    add_files("test_dataset.cpp", "../dataset.cpp")
    add_packages("fmt", "stb", "nlohmann_json", "glm")
    add_tests("default")
    add_tests("args", {runargs = {"--dataset", "data/nerf_synthetic/lego", "--half_res"}})
    add_tests("full_res", {runargs = {"--dataset", "data/nerf_synthetic/lego", "--full_res"}})