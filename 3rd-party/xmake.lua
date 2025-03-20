package("kompute")
    set_homepage("https://github.com/KomputeProject/kompute")
    set_description("General purpose GPU compute framework for cross vendor graphics cards")
    set_license("Apache-2.0")

    add_configs("vk_header_version", {description = "Vulkan header version"})

    set_sourcedir(path.join(os.scriptdir(), "kompute"))

    add_deps("cmake", "vulkan-loader", "fmt")
    on_install(function (package)
        local configs = {}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DKOMPUTE_OPT_BUILD_AS_SHARED_LIB=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DKOMPUTE_OPT_USE_BUILT_IN_FMT=OFF")
        table.insert(configs, "-DKOMPUTE_OPT_USE_BUILT_IN_GOOGLE_TEST=ON" )
        table.insert(configs, "-DKOMPUTE_OPT_USE_BUILT_IN_PYBIND11=ON")
        table.insert(configs, "-DKOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER=ON")
        table.insert(configs, "-DKOMPUTE_OPT_USE_BUILT_IN_SPDLOG=OFF")
        table.insert(configs, "-DKOMPUTE_OPT_LOG_LEVEL=Off")
        table.insert(configs, "-DKOMPUTE_OPT_INSTALL=ON")

        if package:config("vk_header_version") ~= nil then
            local vk_header_version = package:config("vk_header_version")
            table.insert(configs, "-DKOMPUTE_OPT_BUILT_IN_VULKAN_HEADER_TAG=" .. vk_header_version)
        end
        import("package.tools.cmake").install(package, configs)
    end)

    on_test(function (package)
        assert(package:has_cxxtypes("kp::Manager", {includes = "kompute/Kompute.hpp"}))
    end)
package_end()
