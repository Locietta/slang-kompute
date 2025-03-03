add_requires("stb", "nlohmann_json", "glm", "access_private")
add_defines("KOMPUTE_OPT_ACTIVE_LOG_LEVEL=KOMPUTE_LOG_LEVEL_WARN")

target("nerf")
    set_kind("binary")
    add_rules("slang2spv", { bin2c = true, col_major = true })
    add_files("*.cpp", "shaders/*.slang")
    add_packages("kompute", "fmt", "stb", "nlohmann_json", "glm", "access_private")

includes("tests")

