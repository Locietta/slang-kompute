rule("slang2spv")
    set_extensions(".slang")
    on_load(function (target)
        local is_bin2c = target:extraconf("rules", "slang2spv", "bin2c")
        if is_bin2c then 
            local headerdir = path.join(target:autogendir(), "rules", "slang2spv")
            if not os.isdir(headerdir) then 
                os.mkdir(headerdir)
            end
            target:add("includedirs", headerdir)
        end
    end)

    before_buildcmd_file(function (target, batchcmds, sourcefile_slang, opt) 
        import("lib.detect.find_tool")
        
        local slangc = assert(find_tool("slangc"), "slangc not found!")

        -- slang to spv
        local basename = path.basename(sourcefile_slang)

        local targetenv = target:extraconf("rules", "slang2spv", "targetenv") or "vulkan1.0"
        local outputdir = target:extraconf("rules", "slang2spv", "outputdir") or path.join(target:autogendir(), "rules", "slang2spv")
        -- slang default to row major matrix layout
        local col_major = target:extraconf("rules", "slang2spv", "col_major") or false

        local spvfilepath = path.join(outputdir, basename .. ".spv")
        
        -- https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md#-profile
        local profile = target:extraconf("rules", "slang2spv", "profile") or "glsl_460"

        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.slang %s", sourcefile_slang)
        batchcmds:mkdir(outputdir)

        slangc_opt = {
            path(sourcefile_slang),
            "-O2",
            "-target", "spirv",
            "-profile", profile,
            col_major and "-matrix-layout-column-major" or "-matrix-layout-row-major",
            "-o", path(spvfilepath),
            "-entry", "main",
        }

        batchcmds:vrunv(slangc.program, slangc_opt)

        -- bin2c
        local outputfile = spvfilepath
        local is_bin2c = target:extraconf("rules", "slang2spv", "bin2c")

        if is_bin2c then 
            -- get header file
            local headerdir = outputdir
            local headerfile = path.join(headerdir, path.filename(spvfilepath) .. ".h")

            target:add("includedirs", headerdir)
            outputfile = headerfile

            -- add commands
            local argv = {"lua", "private.utils.bin2c", "--nozeroend", "-i", path(spvfilepath), "-o", path(headerfile)}
            batchcmds:vrunv(os.programfile(), argv, {envs = {XMAKE_SKIP_HISTORY = "y"}})
        end 

        batchcmds:add_depfiles(sourcefile_slang)
        batchcmds:set_depmtime(os.mtime(outputfile))
        batchcmds:set_depcache(target:dependfile(outputfile))
    end)

    after_clean(function (target, batchcmds, sourcefile_slang) 
        import("private.action.clean.remove_files")

        local outputdir = target:extraconf("rules", "slang2spv", "outputdir") or path.join(target:targetdir(), "shader")
        remove_files(path.join(outputdir, "*.spv"))
        remove_files(path.join(outputdir, "*.spv.h"))
    end)