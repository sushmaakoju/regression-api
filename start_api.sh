echo "this batch file will install requirements and start flask api"
echo "check python version is 3x"
pyv="$(python -V 3>&1)"
echo "$pyv"
echo "check current directory"
BASEDIR=$(dirname "$0")
echo $BASEDIR
pip install -r requirements.txt
python run.py