set_xmakever("2.9.7")
set_project("slang-kompute")

add_rules("mode.release", "mode.debug", "mode.releasedbg")

set_languages("cxx23")

if is_os("windows") then
    set_toolchains("clang-cl")
    add_defines("_CRT_SECURE_NO_WARNINGS")
end

-- add_repositories("local-repo xmake")
-- add_requires("kompute 0.9.*", "fmt")

includes("3rd-party")

add_moduledirs("xmake/modules")
includes("xmake/*.lua")

add_requires("kompute")
-- add_requires("kompute", {debug = true})
add_requires("fmt")

includes("compute")
