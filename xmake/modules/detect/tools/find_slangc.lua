-- imports
import("lib.detect.find_program")
import("lib.detect.find_programver")

-- find slangc
--
-- @param opt   the argument options, e.g. {version = true, program = "c:\xxx\slangc.exe"}
--
-- @return      program, version
--
-- @code
--
-- local slangc = find_slangc()
-- local slangc, version = find_slangc({version = true})
-- local slangc, version = find_slangc({version = true, program = "c:\xxx\slangc.exe"})
--
-- @endcode
--
function main(opt)
    opt = opt or {}
    opt.paths = opt.paths or {
        "$(env PATH)",
        "$(env VULKAN_SDK)/Bin",
        "$(env VK_SDK_PATH)/Bin",
    }
    opt.check = opt.check or "-v"
    local program = find_program(opt.program or "slangc.exe", opt)
    local version = nil
    if program and opt and opt.version then
        version = find_programver(program, opt)
    end
    return program, version
end