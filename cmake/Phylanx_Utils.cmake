# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakeParseArguments)
include(Phylanx_Include)

phylanx_include(
  Message
  Option
  AddCompileFlag
  AddDefinitions
  AddExecutable
  AddLibraryHeaders
  AddLibrarySources
  AddLinkFlag
  AddPseudoDependencies
  AddPseudoTarget
  AddSourceGroup
  AddTest
  AddPythonTest
  AppendProperty
  CreateSymbolicLink
  DetectCppDialect
  Documentation
  ExportTargets
  ForceOutOfTreeBuild
  GetPythonExtensionLocation
  HandleComponentDependencies
  HighFive
  IsTarget
  SetCMakePolicy
  SetupCompilerFlags
  SetOutputPaths
  SetupTarget
  ShortenPseudoTarget
)

phylanx_include(
  SetupHPX
  SetupBlaze
  SetupPybind11
)
