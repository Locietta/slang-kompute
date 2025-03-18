add_defines("KOMPUTE_OPT_ACTIVE_LOG_LEVEL=KOMPUTE_LOG_LEVEL_WARN")
add_requires("glm")

target("reduce")
    set_kind("binary")
    add_files("*.cpp", "shaders/*.slang")
    add_packages("fmt", "kompute", "glm")
    add_rules("slang2spv", { bin2c = true, col_major = true })