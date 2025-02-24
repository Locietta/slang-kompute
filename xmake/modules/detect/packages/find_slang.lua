-- imports
import("lib.detect.find_path")
import("lib.detect.find_library")
import("lib.detect.find_tool")

function main(opt)
    local slangc = assert(find_tool("slangc"), "slang not found!")
    local slang_dir = path.directory(path.directory(slangc.program))

    local bin_dir = path.join(slang_dir, "bin")
    local link_dir = path.join(slang_dir, "lib")
    local includedirs = { path.join(slang_dir, "include") }
    local libfiles = {}

    if opt.plat == "windows" then
        -- collect all dll files in bin folder
        for _, file in ipairs(os.files(path.join(bin_dir, "*.dll"))) do
            table.insert(libfiles, file)
        end
        -- collect all lib files in lib folder
        for _, file in ipairs(os.files(path.join(link_dir, "*.lib"))) do
            table.insert(libfiles, path.join(link_dir, file))
        end
        links = {
            "slang",
            "slang-rt",
            "gfx",
        }
    else
        -- linux

        -- collect all so files in lib folder
        for _, file in ipairs(os.files(path.join(link_dir, "*.so"))) do
            table.insert(libfiles, path.join(link_dir, file))
        end

        table.insert(includedirs, path.join(slang_dir, "include/shader-slang"))

        links = {
            "slang",
            "slang-rt",
            "slang-glslang",
            "slang-glsl-module",
            "libslang-llvm",
            "gfx",
        }
    end

    return {
        linkdirs = { link_dir },
        includedirs = includedirs,
        libfiles = libfiles,
        links = links,
        bindir = bin_dir,
        shared = true,
    }
end