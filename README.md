# slang-kompute

Example of using [slang](https://shader-slang.org/) shaders with [kompute](https://github.com/KomputeProject/kompute) and [xmake](https://xmake.io/#/).

## Prerequisites

### Windows

Using [scoop](https://scoop.sh) to install dependencies is recommended:

```
scoop install xmake llvm slang cmake
```

You will also need to install Visual Studio or VS Build Tools.

### Linux

Install dependencies with your distro's package manager. For example, on Arch:

```
paru -Syu xmake clang llvm cmake shader-slang-bin
```

> `shader-slang-bin` is on aur, so you will need an aur helper (not necessarily `paru` though). Alternatively, you can download slang binaries from the [slang repo](https://github.com/shader-slang/slang/releases/latest) and use `pacman` for other dependencies.

<details>
  <summary>WSL 2 (At your own risk)</summary>

### WSL 2

WSL 2 does not officially support Vulkan. But Microsoft does provide an experimental WSL2 vulkan driver (dozen) in mesa, which implements vulkan on top of DirectX 12.

For Ubuntu, there's a ppa that already builds mesa with dozen enabled for you:
```
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade
```

For Arch, you can use this [PKGBUILD](https://gist.github.com/Locietta/c4ed577e44c444679dabebb3732901e3) to build mesa with dozen enabled. 

**NOTE**: dozen is very experimental and has conformances issues (incomplete API and differences in behavior). Also it only supports vulkan <=1.2.296. To make it work, you will have to change kompute requires in root `xmake.lua` to:

```
add_requires("kompute",{ configs = { vk_header_version = "v1.2.203" } })
```
</details>

## Build

1. Clone this repo with submodules:

```
git clone --recursive https://github.com/Locietta/slang-kompute.git
```

2. Build with xmake:

```
xmake f -c && xmake
```

## Run

```
xmake r compute
```