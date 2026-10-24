@echo off
setlocal
rem ---------------------------------------------------------------------------
rem Trains and evaluates the models of the project (Windows; build.sh is the
rem Linux and macOS version).
rem
rem Usage:
rem   scripts\build.bat              train the decision tree, logistic regression and neural network
rem   scripts\build.bat tree         train one pair model: tree, logistic, neural
rem   scripts\build.bat retrieval    train the retrieval (embedding) model
rem   scripts\build.bat difficulty   train the difficulty detector (data\models\difficulty.ser)
rem   scripts\build.bat calibrate    choose the threshold of every difficulty level for every saved
rem                                  model (NAME.ser.levels); needs no retraining
rem   scripts\build.bat evaluate     compare pooled, true-level and detected-level thresholds on the
rem                                  held-out half
rem   scripts\build.bat check        run the self-checks (gradients, metrics, features, save and load)
rem   scripts\build.bat clean        delete the compiled classes in out\ and stop
rem
rem What a training run does (cli.Main, cli.RetrievalMain):
rem   - It trains on all three difficulty levels together, the same way the web app's Train page does.
rem   - The feature settings come from features\FeatureConfig.java and can be changed with -D options,
rem     for example -Dchar.n=1,2,3 -Dchar.buckets=512 -Dcasing=lower.
rem   - The decision threshold is tuned on one half of the PAN validation split. The final numbers
rem     are measured on the other half, which the model never saw during training or threshold selection.
rem   - The model is saved under data\models\ and a row is appended to data\model-metrics.json.
rem     The web app reads both, so the run also shows up on its Results page.
rem
rem JVM options are passed through the JAVA_OPTS environment variable:
rem   set JAVA_OPTS=-Xmx16g
rem   set JAVA_OPTS=-Dsubset=50000          quick trial run on a subset of the training pairs
rem   set JAVA_OPTS=-Ddifficulty=hard       train and score one difficulty level only
rem
rem The JDK is taken from ASD_JDK_HOME (default C:\jdk-25 when that folder exists), otherwise
rem from JAVA_HOME or the PATH.
rem ---------------------------------------------------------------------------

set "COMMON=%~dp0lib\common.bat"

set "TARGET=%~1"
if not defined TARGET set "TARGET=all"

for %%T in (all tree logistic neural retrieval difficulty calibrate evaluate check clean) do (
    if /I "%TARGET%"=="%%T" goto :target_valid
)
echo ERROR: unknown target "%TARGET%"
echo Use: all ^| tree ^| logistic ^| neural ^| retrieval ^| difficulty ^| calibrate ^| evaluate ^| check ^| clean
goto :failed

:target_valid

rem "clean" needs neither Java nor a compilation.
if /I "%TARGET%"=="clean" goto :clean

call "%COMMON%" :configure_java
call "%COMMON%" :require_jdk
if errorlevel 1 goto :failed
call "%COMMON%" :enter_project_root
if errorlevel 1 goto :failed
call "%COMMON%" :warn_if_dataset_missing

if not exist "%ASD_MODELS_DIR%\" mkdir "%ASD_MODELS_DIR%"
call "%COMMON%" :clean_classes
if errorlevel 1 goto :failed
call "%COMMON%" :compile cli\Main.java cli\RetrievalMain.java cli\DifficultyMain.java cli\SelfCheck.java cli\TiraRunner.java
if errorlevel 1 goto :failed

rem Every target is one entry point of the cli package. The pair models share
rem cli.Main and receive their name as the argument.
set "ENTRY=cli.Main"
set "ENTRY_ARGS=%TARGET%"
if /I "%TARGET%"=="retrieval"  set "ENTRY=cli.RetrievalMain" & set "ENTRY_ARGS="
if /I "%TARGET%"=="difficulty" set "ENTRY=cli.DifficultyMain" & set "ENTRY_ARGS=train"
if /I "%TARGET%"=="calibrate"  set "ENTRY=cli.DifficultyMain" & set "ENTRY_ARGS=calibrate"
if /I "%TARGET%"=="evaluate"   set "ENTRY=cli.DifficultyMain" & set "ENTRY_ARGS=evaluate"
if /I "%TARGET%"=="check"      set "ENTRY=cli.SelfCheck"      & set "ENTRY_ARGS="

echo.
echo Running: %TARGET%
echo.
java %JAVA_OPTS% -cp "%ASD_OUT%" %ENTRY% %ENTRY_ARGS%
if errorlevel 1 (
    echo ERROR: the program stopped with an error.
    goto :failed
)

echo.
echo Finished: %TARGET%
if /I "%TARGET%"=="check"    goto :succeeded
if /I "%TARGET%"=="evaluate" goto :succeeded
echo Trained models and thresholds are under: %ASD_MODELS_DIR%
echo Metrics are appended to:                 %ASD_METRICS_FILE%
echo The web app's Results page shows them.
goto :succeeded

:clean
call "%COMMON%" :enter_project_root
if errorlevel 1 goto :failed
echo Removing %ASD_OUT%
if exist "%ASD_OUT%\" rd /s /q "%ASD_OUT%"
if exist "%ASD_OUT%\" (
    echo ERROR: could not remove the out folder.
    goto :failed
)
echo Done.
goto :succeeded

:failed
echo.
echo BUILD FAILED
exit /b 1

:succeeded
exit /b 0
