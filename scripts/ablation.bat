
@echo off
setlocal

REM ============================================================
REM HYP - AUTOMATED ABLATION STUDY
REM ============================================================
REM Feature-group importance:
REM Importance(G) = F1_all - F1_without_G
REM
REM Supports feature-group and parameter ablation for:
REM Decision Tree, Logistic Regression and Neural Network.
REM
REM Usage:
REM   scripts\ablation.bat
REM   scripts\ablation.bat all
REM   scripts\ablation.bat features
REM   scripts\ablation.bat params
REM
REM Optional JVM options:
REM   set JAVA_OPTS=-Xmx24g
REM   scripts\ablation.bat
REM
REM   set JAVA_OPTS=-Dmodels=tree
REM   scripts\ablation.bat params
REM
REM   set JAVA_OPTS=-Dseeds=42,123,456
REM   scripts\ablation.bat
REM
REM   set JAVA_OPTS=-Dquick=true -Dsubset=50000
REM   scripts\ablation.bat
REM
REM Modes: all | features | params | pipeline (n-gram order 1..5, buckets, casing, pair representation)
REM        | retrieval (the retrieval model, writes retrieval_ablation.csv)
REM
REM Results:
REM   data\ablation\ablation_results.csv
REM   data\ablation\ablation_report.md
REM   data\ablation\ablation_tables.tex

cls

REM ============================================================
REM JAVA CONFIGURATION
REM Keep the same Java installation as build.bat.
REM ============================================================

set "JAVA_HOME_OVERRIDE=C:\jdk-25"

if exist "%JAVA_HOME_OVERRIDE%\bin\javac.exe" (
    set "JAVA_HOME=%JAVA_HOME_OVERRIDE%"
    set "PATH=%JAVA_HOME%\bin;%PATH%"
)

echo.
echo        HYP - ABLATION STUDY
echo.
echo JAVA_HOME:
if defined JAVA_HOME (
    echo %JAVA_HOME%
) else (
    echo ^<not set, relying on PATH^>
)
echo.

REM ============================================================
REM MODE SELECTION
REM ============================================================

set "MODE=%~1"
if not defined MODE set "MODE=all"

if /I "%MODE%"=="all" goto MODE_VALID
if /I "%MODE%"=="features" goto MODE_VALID
if /I "%MODE%"=="params" goto MODE_VALID
if /I "%MODE%"=="pipeline" goto MODE_VALID
if /I "%MODE%"=="retrieval" goto MODE_VALID

echo ERROR: Unknown mode "%MODE%"
echo Use: all ^| features ^| params
goto ERROR

:MODE_VALID

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
REM PROJECT ROOT AND PATHS
REM ============================================================

cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: Could not move to project directory.
    goto ERROR
)

echo Project directory:
cd
echo.

set "SRC=%CD%\src\main\java"
set "OUT=%CD%\out"
set "DATASET_DIR=%CD%\dataset\mawsa26-pan-zenodo-DATA"

if not exist "%SRC%\Ablation.java" (
    echo ERROR: Ablation.java not found at:
    echo %SRC%\Ablation.java
    goto ERROR
)

if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%" (
    echo ERROR: Could not create output directory.
    goto ERROR
)

REM ============================================================
REM DATASET CHECK
REM ============================================================

if not exist "%DATASET_DIR%\" (
    echo WARNING: Dataset folder not found at:
    echo   %DATASET_DIR%
    echo.
)

REM ============================================================
REM COMPILE ABLATION
REM javac follows imports through -sourcepath.
REM ============================================================

echo          ^> Compiling Ablation.java
echo.

javac -encoding UTF-8 -sourcepath "%SRC%" -d "%OUT%" "%SRC%\Ablation.java" "%SRC%\RetrievalMain.java"

if errorlevel 1 (
    echo ERROR: Compilation failed.
    goto ERROR
)

echo.
echo Compilation successful.
echo.

REM ============================================================
REM RUN ABLATION STUDY
REM ============================================================

echo          ^> Running the ablation study: %MODE%
echo.

if /I "%MODE%"=="retrieval" (
    if not exist "data\ablation" mkdir "data\ablation"
    java %JAVA_OPTS% -Dret.csv=data/ablation/retrieval_ablation.csv -cp "%OUT%" RetrievalMain ablate
) else (
    java %JAVA_OPTS% -cp "%OUT%" Ablation "%MODE%"
)

if errorlevel 1 (
    echo ERROR: Ablation study failed.
    goto ERROR
)

echo.
echo        ^> Ablation study finished.
echo        ^> Report: %CD%\data\ablation\ablation_report.md
echo        ^> CSV:    %CD%\data\ablation\ablation_results.csv
echo        ^> LaTeX:  %CD%\data\ablation\ablation_tables.tex
echo.

goto SUCCESS

REM ============================================================
REM EXIT HANDLING
REM ============================================================

:ERROR
echo.
echo              ^> ABLATION FAILED
echo.
endlocal
exit /b 1

:SUCCESS
endlocal
exit /b 0
