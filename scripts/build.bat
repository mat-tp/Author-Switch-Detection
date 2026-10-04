@echo off
cls

REM Tshephang Matlala
REM HYP TRAINING ML MODELS
REM Trains and compares all the models (Decision Tree, Logistic Regression,
REM Neural Network) using ONLY javac/java - no Maven, no internet access and
REM no IDE needed. The Spring Boot web app (mvnw.cmd) is a separate,
REM optional thing; this script is for training/testing the models straight
REM from the command line.
REM
REM Usage:
REM   scripts\build.bat              runs all the models (train + validation + test)
REM   scripts\build.bat tree         runs only one: tree | logistic | neural
REM   scripts\build.bat clean        deletes out\ and exits (nothing is run)
REM
REM Version for Windows Systems

setlocal enabledelayedexpansion

REM JAVA CONFIGURATION
REM Set JAVA_HOME below if Java is not already on PATH / JAVA_HOME.
REM Leave it blank to just use whatever "java"/"javac" are already on PATH.
set "JAVA_HOME_OVERRIDE=C:\jdk-25"

if exist "%JAVA_HOME_OVERRIDE%\bin\javac.exe" (
    set "JAVA_HOME=%JAVA_HOME_OVERRIDE%"
    set "PATH=%JAVA_HOME%\bin;%PATH%"
)

echo.
echo        "> HYP - Project"
echo.

echo JAVA_HOME:
if defined JAVA_HOME (
    echo %JAVA_HOME%
) else (
    echo ^<not set, relying on PATH^>
)
echo.

REM WHICH MODEL(S)

REM No argument = all models
REM Available: all | tree | logistic | neural | clean

if "%~1"=="" (
    set "MODEL=all"
) else (
    set "MODEL=%~1"
)

if /I "%MODEL%"=="clean" (
    cd /d "%~dp0.."
    echo Removing "%CD%\out" ...
    if exist "out" rd /s /q "out"
    echo Done.
    goto END
)

if /I "%MODEL%"=="all" goto MODEL_VALID
if /I "%MODEL%"=="tree" goto MODEL_VALID
if /I "%MODEL%"=="logistic" goto MODEL_VALID
if /I "%MODEL%"=="neural" goto MODEL_VALID

echo.
echo ERROR: Unknown model "%MODEL%"
echo Use: all ^| tree ^| logistic ^| neural ^| clean
goto ERROR

:MODEL_VALID

REM CHECK JAVA
echo ~~~ Checking Java ~~~

javac -version
if errorlevel 1 (
    echo.
    echo ERROR: javac could not be found.
    echo Install a JDK ^(17+^) and either add it to PATH or set JAVA_HOME_OVERRIDE above.
    goto ERROR
)

java -version
if errorlevel 1 (
    echo.
    echo ERROR: java could not be found.
    goto ERROR
)

echo.
echo Java is ready.
echo.

REM MOVE TO PROJECT ROOT (this script lives in <project root>\scripts)
cd /d "%~dp0.."

echo Project directory:
cd
echo.

REM PROJECT PATHS
REM Main.java (plus the data / features / models packages it uses) lives
REM under the normal Maven source folder, so this script and a Maven build
REM compile from exactly the same source of truth.
set "SRC=%CD%\src\main\java"
set "OUT=%CD%\out"
set "MODELS_DIR=%CD%\data\models"
set "METRICS_FILE=%CD%\data\model-metrics.json"
set "DATASET_DIR=%CD%\dataset\mawsa26-pan-zenodo-DATA"

REM CLEAN
echo              ^> Cleaning previously compiled classes
echo.

if exist "%OUT%" (
    del /S /Q "%OUT%\*.class" >nul 2>&1
) else (
    mkdir "%OUT%"
)

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo Cleaning complete.
echo.

REM DATASET CHECK (a friendly warning instead of a wall of "directory does
REM not exist" messages from inside DatasetLoader)
if not exist "%DATASET_DIR%" (
    echo WARNING: Dataset folder not found at:
    echo   %DATASET_DIR%
    echo Training will fail immediately with "No training data was loaded."
    echo Place the MAWSA dataset there first ^(see README.md^).
    echo.
)

REM COMPILE
REM Only Main.java is passed to javac; javac follows its imports (data.*,
REM features.*, models.*) automatically via -sourcepath, without touching
REM the Spring Boot (com.hyp.*) classes, so no Spring/Maven dependencies are
REM needed on the classpath.
echo          ^> Compiling Java Source Code
echo.

javac ^
    -sourcepath "%SRC%" ^
    -d "%OUT%" ^
    "%SRC%\Main.java"

if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed.
    goto ERROR
)

echo.
echo Compilation successful.
echo.

REM RUN
REM Main trains on ALL 3 difficulty splits combined (same as RealTrainer, used
REM by the Spring web app's Train page), with the 11 compressed features (see
REM USE_FULL_FEATURES in Main.java). It tunes the decision threshold on one
REM half of the validation split, and reports final numbers on the OTHER half
REM (the held-out "test" half), which the model never saw during training or
REM thresholding. The trained model is saved under data\models\, and a row is
REM appended to data\model-metrics.json - the SAME files the Spring Boot web
REM app (mvnw.cmd spring-boot:run) reads, so this run shows up there too.
REM
REM Extra JVM options can be given with JAVA_OPTS, for example:
REM   set JAVA_OPTS=-Xmx8g
REM   set JAVA_OPTS=-Dsubset=50000
REM   scripts\build.bat all
echo          ^> Running ML Model(s):  %MODEL%
echo.

java %JAVA_OPTS% -cp "%OUT%" Main "%MODEL%"

if errorlevel 1 (
    echo.
    echo ERROR: Program execution failed.
    goto ERROR
)

echo.
echo        ^> ML Training Finished
echo        ^> Trained model(s) saved under: %MODELS_DIR%
echo        ^> Metrics appended to:          %METRICS_FILE%
echo        ^> (open the web app's Results page to see them)
echo.

goto END

:ERROR

echo.
echo              ^> BUILD FAILED
echo.

:END

pause

endlocal
exit /b 0
