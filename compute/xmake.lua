
target("compute")
    set_kind("binary")
    add_files("*.cpp", "shaders/*.slang")
    add_packages("kompute", "fmt")
    add_rules("slang2spv", { bin2c = true, col_major = true })