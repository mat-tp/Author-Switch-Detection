@echo off
rem ---------------------------------------------------------------------------
rem Shared helpers of the project's batch files (build.bat and ablation.bat).
rem
rem This file is called, never run directly. The first argument is the name of
rem the helper, the remaining arguments belong to that helper:
rem
rem     call "%~dp0lib\common.bat" :configure_java
rem     call "%~dp0lib\common.bat" :require_jdk
rem     call "%~dp0lib\common.bat" :enter_project_root
rem     call "%~dp0lib\common.bat" :warn_if_dataset_missing
rem     call "%~dp0lib\common.bat" :clean_classes
rem     call "%~dp0lib\common.bat" :compile cli\Main.java cli\SelfCheck.java
rem
rem Every helper returns exit code 0 on success and 1 on failure, so the caller
rem checks it with "if errorlevel 1". The helpers print their own error message.
rem The calling batch file should use SETLOCAL, because the helpers set variables
rem in the caller's environment:
rem
rem     ASD_ROOT          project root
rem     ASD_SRC           Java source root (the same folder Maven compiles)
rem     ASD_OUT           folder for the compiled classes of the command line tools
rem     ASD_MODELS_DIR    folder for the trained models
rem     ASD_METRICS_FILE  json file the web app reads for its Results page
rem     ASD_DATASET_DIR   folder with the PAN 2026 MAWSA data
rem ---------------------------------------------------------------------------
goto %~1


rem Chooses the JDK. ASD_JDK_HOME may name the JDK folder; when it is not set,
rem C:\jdk-25 is used if it exists. Otherwise JAVA_HOME or the PATH decides.
:configure_java
if not defined ASD_JDK_HOME set "ASD_JDK_HOME=C:\jdk-25"
if not exist "%ASD_JDK_HOME%\bin\javac.exe" goto :configure_java_report
set "JAVA_HOME=%ASD_JDK_HOME%"
set "PATH=%JAVA_HOME%\bin;%PATH%"
:configure_java_report
if not defined JAVA_HOME goto :configure_java_unset
echo JAVA_HOME: %JAVA_HOME%
exit /b 0
:configure_java_unset
echo JAVA_HOME: not set, relying on the PATH
exit /b 0


rem Checks that javac and java can be found and prints their versions.
:require_jdk
where javac >nul 2>&1
if errorlevel 1 goto :require_jdk_missing
where java >nul 2>&1
if errorlevel 1 goto :require_jdk_missing
javac -version
if errorlevel 1 exit /b 1
java -version 2>&1 | findstr /B /C:"openjdk version" /C:"java version"
exit /b 0
:require_jdk_missing
echo ERROR: javac or java was not found.
echo Install a JDK (21 or newer, the web app needs 25) and add it to the PATH,
echo or set ASD_JDK_HOME to the folder of the JDK.
exit /b 1


rem Changes into the project root (the parent folder of scripts) and defines the path variables.
:enter_project_root
cd /d "%~dp0..\.."
if errorlevel 1 goto :enter_project_root_failed
set "ASD_ROOT=%CD%"
set "ASD_SRC=%CD%\src\main\java"
set "ASD_OUT=%CD%\out"
set "ASD_MODELS_DIR=%CD%\data\models"
set "ASD_METRICS_FILE=%CD%\data\model-metrics.json"
set "ASD_DATASET_DIR=%CD%\dataset\mawsa26-pan-zenodo-DATA"
echo Project directory: %ASD_ROOT%
exit /b 0
:enter_project_root_failed
echo ERROR: could not move to the project directory.
exit /b 1


rem Warns when the PAN data is not where the programs expect it.
:warn_if_dataset_missing
if exist "%ASD_DATASET_DIR%\" exit /b 0
echo WARNING: the dataset folder was not found at:
echo   %ASD_DATASET_DIR%
echo   Training and the ablation study need it, see README.md section 6.
exit /b 0


rem Deletes the class files of an earlier compilation, so that no stale class
rem survives a renamed or removed source file, and makes sure out\ exists.
:clean_classes
if not exist "%ASD_OUT%\" goto :clean_classes_create
del /S /Q "%ASD_OUT%\*.class" >nul 2>&1
exit /b 0
:clean_classes_create
mkdir "%ASD_OUT%"
exit /b %ERRORLEVEL%


rem Compiles the given programs into %ASD_OUT%. javac follows their imports
rem through -sourcepath, so only the entry points are listed, relative to
rem src\main\java. The Spring Boot classes are never reached, which is why no
rem Maven dependency is needed.
:compile
shift
set "ASD_SOURCES="
:compile_collect
if "%~1"=="" goto :compile_run
if exist "%ASD_SRC%\%~1" goto :compile_found
echo ERROR: source file not found: %ASD_SRC%\%~1
exit /b 1
:compile_found
set ASD_SOURCES=%ASD_SOURCES% "%ASD_SRC%\%~1"
shift
goto :compile_collect
:compile_run
if not exist "%ASD_OUT%\" mkdir "%ASD_OUT%"
echo Compiling the command line programs
javac -encoding UTF-8 -sourcepath "%ASD_SRC%" -d "%ASD_OUT%" %ASD_SOURCES%
if errorlevel 1 goto :compile_failed
echo Compilation successful.
exit /b 0
:compile_failed
echo ERROR: compilation failed.
exit /b 1
