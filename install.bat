@ECHO OFF
ECHO Installing required packages...

REM Setup virtualenv
pip install virtualenv
virtualenv env

REM Install packages
env\Scripts\pip3 install -r requirements.txt

ECHO Done.

PAUSE