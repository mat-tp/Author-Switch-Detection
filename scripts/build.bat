
@echo off
setlocal

REM Tshephang Matlala
REM HYP TRAINING ML MODELS
REM Trains and compares all models using javac/java.
REM
REM Usage:
REM   scripts\build.bat
REM   scripts\build.bat all
REM   scripts\build.bat tree
REM   scripts\build.bat logistic
REM   scripts\build.bat neural
REM   scripts\build.bat clean

cls

REM ============================================================
REM JAVA CONFIGURATION - PRESERVED
REM ============================================================

set "JAVA_HOME_OVERRIDE=C:\jdk-25"

if exist "%JAVA_HOME_OVERRIDE%\bin\javac.exe" (
    set "JAVA_HOME=%JAVA_HOME_OVERRIDE%"
    set "PATH=%JAVA_HOME%\bin;%PATH%"
)

echo.
echo        HYP - Project
echo.
echo JAVA_HOME:
if defined JAVA_HOME (
    echo %JAVA_HOME%
) else (
    echo ^<not set, relying on PATH^>
)
echo.

REM ============================================================
REM MODEL SELECTION
REM ============================================================

set "MODEL=%~1"
if not defined MODEL set "MODEL=all"

if /I "%MODEL%"=="clean" goto CLEAN

if /I "%MODEL%"=="all" goto MODEL_VALID
if /I "%MODEL%"=="tree" goto MODEL_VALID
if /I "%MODEL%"=="logistic" goto MODEL_VALID
if /I "%MODEL%"=="neural" goto MODEL_VALID
if /I "%MODEL%"=="retrieval" goto MODEL_VALID
if /I "%MODEL%"=="difficulty" goto MODEL_VALID
if /I "%MODEL%"=="calibrate" goto MODEL_VALID
if /I "%MODEL%"=="evaluate" goto MODEL_VALID
if /I "%MODEL%"=="check" goto MODEL_VALID

echo ERROR: Unknown model "%MODEL%"
echo Use: all ^| tree ^| logistic ^| neural ^| retrieval ^| difficulty ^| calibrate ^| evaluate ^| check ^| clean
goto ERROR

:MODEL_VALID

REM ============================================================
REM JAVA CHECKS
REM ============================================================

echo ~~~ Checking Java ~~~

where javac >nul 2>&1
if errorlevel 1 (
    echo ERROR: javac could not be found.
    echo Install a JDK 17+ or check JAVA_HOME_OVERRIDE and PATH.
    goto ERROR
)

where java >nul 2>&1
if errorlevel 1 (
    echo ERROR: java could not be found.
    goto ERROR
)

echo Java compiler version:
javac -version
if errorlevel 1 goto ERROR

echo.
echo Java runtime version:
java -version
if errorlevel 1 goto ERROR

echo.
echo Java is ready.
echo.

REM ============================================================
REM PROJECT ROOT
REM ============================================================

cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: Could not move to project directory.
    goto ERROR
)

echo Project directory:
cd
echo.

REM ============================================================
REM PROJECT PATHS
REM ============================================================

set "SRC=%CD%\src\main\java"
set "OUT=%CD%\out"
set "MODELS_DIR=%CD%\data\models"
set "METRICS_FILE=%CD%\data\model-metrics.json"
set "DATASET_DIR=%CD%\dataset\mawsa26-pan-zenodo-DATA"

if not exist "%SRC%\Main.java" (
    echo ERROR: Main.java not found at:
    echo %SRC%\Main.java
    goto ERROR
)

REM ============================================================
REM CLEAN PREVIOUS COMPILED CLASSES
REM ============================================================

echo          ^> Cleaning previously compiled classes
echo.

if exist "%OUT%" (
    del /S /Q "%OUT%\*.class" >nul 2>&1
) else (
    mkdir "%OUT%"
    if errorlevel 1 goto ERROR
)

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%MODELS_DIR%" (
    echo ERROR: Could not create models directory.
    goto ERROR
)

echo Cleaning complete.
echo.

REM ============================================================
REM DATASET CHECK
REM ============================================================

if not exist "%DATASET_DIR%\" (
    echo WARNING: Dataset folder not found at:
    echo   %DATASET_DIR%
    echo Training may fail because no training data is loaded.
    echo Place the MAWSA dataset there first.
    echo.
)

REM ============================================================
REM COMPILE
REM ============================================================

echo          ^> Compiling Java Source Code
echo.

javac -encoding UTF-8 -sourcepath "%SRC%" -d "%OUT%" "%SRC%\Main.java" "%SRC%\RetrievalMain.java" "%SRC%\SelfCheck.java" "%SRC%\TiraRunner.java" "%SRC%\DifficultyMain.java"

if errorlevel 1 (
    echo ERROR: Compilation failed.
    goto ERROR
)

echo.
echo Compilation successful.
echo.

REM ============================================================
REM RUN MODEL(S)
REM ============================================================

echo          ^> Running ML Model(s): %MODEL%
echo.

if /I "%MODEL%"=="retrieval" (
    java %JAVA_OPTS% -cp "%OUT%" RetrievalMain
) else if /I "%MODEL%"=="difficulty" (
    java %JAVA_OPTS% -cp "%OUT%" DifficultyMain train
) else if /I "%MODEL%"=="calibrate" (
    java %JAVA_OPTS% -cp "%OUT%" DifficultyMain calibrate
) else if /I "%MODEL%"=="evaluate" (
    java %JAVA_OPTS% -cp "%OUT%" DifficultyMain evaluate
) else if /I "%MODEL%"=="check" (
    java %JAVA_OPTS% -cp "%OUT%" SelfCheck
) else (
    java %JAVA_OPTS% -cp "%OUT%" Main "%MODEL%"
)

if errorlevel 1 (
    echo ERROR: Program execution failed.
    goto ERROR
)

echo.
echo        ^> ML Training Finished
echo        ^> Trained model(s) saved under: %MODELS_DIR%
echo        ^> Metrics appended to:          %METRICS_FILE%
echo        ^> Open the web app Results page to view the results.
echo.

goto SUCCESS

REM ============================================================
REM CLEAN COMMAND
REM ============================================================

:CLEAN

cd /d "%~dp0.."
if errorlevel 1 goto ERROR

echo Removing "%CD%\out" ...
if exist "out" rd /s /q "out"

if exist "out" (
    echo ERROR: Could not remove out directory.
    goto ERROR
)

echo Done.
goto SUCCESS

REM ============================================================
REM EXIT HANDLING
REM ============================================================

:ERROR
echo.
echo              ^> BUILD FAILED
echo.
endlocal
exit /b 1

:SUCCESS
endlocal
exit /b 0
