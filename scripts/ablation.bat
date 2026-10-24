@echo off
setlocal
rem ---------------------------------------------------------------------------
rem Runs the automated ablation study (Windows; ablation.sh is the Linux and
rem macOS version).
rem
rem The study measures which feature groups and settings matter:
rem   - Feature-group importance:  Importance(G) = F1(all features) - F1(all features without G)
rem   - Parameter ablation of the decision tree, the logistic regression and the neural network.
rem
rem Usage:
rem   scripts\ablation.bat                features + parameters + pipeline, all three models
rem   scripts\ablation.bat features       feature-group ablation only
rem   scripts\ablation.bat params         parameter ablation only
rem   scripts\ablation.bat pipeline       pipeline settings: character n-gram order 1..5, bucket sizes,
rem                                       casing, POS n-grams, pair representation, z-score clipping
rem                                       and the POS tagger
rem   scripts\ablation.bat retrieval      the retrieval model: embedding size, margin, hard-negative
rem                                       mining, n-gram orders, buckets, casing and window
rem
rem Options are passed through the JAVA_OPTS environment variable:
rem   set JAVA_OPTS=-Xmx24g                          more memory
rem   set JAVA_OPTS=-Dmodels=tree                    one model only
rem   set JAVA_OPTS=-Dseeds=42,123,456               mean and standard deviation over three seeds
rem   set JAVA_OPTS=-Dquick=true -Dsubset=50000      fast trial run
rem   set JAVA_OPTS=-Ddifficulty=hard                the whole study on one level (easy, medium, hard)
rem
rem The retrieval study writes to data\ablation\retrieval_ablation.csv; set RET_CSV to choose another file.
rem
rem Results are written while the study runs, so an interrupted run keeps what has finished:
rem   data\ablation\ablation_results.csv   one row per run
rem   data\ablation\ablation_report.md     importance ranking, parameter tables and the best settings
rem   data\ablation\ablation_tables.tex    LaTeX tables for the manuscript
rem
rem The JDK is taken from ASD_JDK_HOME (default C:\jdk-25 when that folder exists), otherwise
rem from JAVA_HOME or the PATH.
rem ---------------------------------------------------------------------------

set "COMMON=%~dp0lib\common.bat"

set "MODE=%~1"
if not defined MODE set "MODE=all"

for %%M in (all features params pipeline retrieval) do (
    if /I "%MODE%"=="%%M" goto :mode_valid
)
echo ERROR: unknown mode "%MODE%"
echo Use: all ^| features ^| params ^| pipeline ^| retrieval
goto :failed

:mode_valid

call "%COMMON%" :configure_java
call "%COMMON%" :require_jdk
if errorlevel 1 goto :failed
call "%COMMON%" :enter_project_root
if errorlevel 1 goto :failed
call "%COMMON%" :warn_if_dataset_missing

rem cli.Ablation reuses the data pipeline of the other programs; javac follows the imports.
call "%COMMON%" :compile cli\Ablation.java cli\RetrievalMain.java
if errorlevel 1 goto :failed

echo.
echo Running the ablation study: %MODE%
echo.

if /I "%MODE%"=="retrieval" goto :run_retrieval
java %JAVA_OPTS% -cp "%ASD_OUT%" cli.Ablation %MODE%
goto :run_finished

:run_retrieval
if not defined RET_CSV set "RET_CSV=data/ablation/retrieval_ablation.csv"
if not exist "%ASD_ROOT%\data\ablation\" mkdir "%ASD_ROOT%\data\ablation"
java %JAVA_OPTS% "-Dret.csv=%RET_CSV%" -cp "%ASD_OUT%" cli.RetrievalMain ablate

:run_finished
if errorlevel 1 (
    echo ERROR: the ablation study stopped with an error.
    goto :failed
)

echo.
echo Finished. Results are in %ASD_ROOT%\data\ablation\
echo   ablation_report.md, ablation_results.csv, ablation_tables.tex, retrieval_ablation.csv
goto :succeeded

:failed
echo.
echo ABLATION FAILED
exit /b 1

:succeeded
exit /b 0
