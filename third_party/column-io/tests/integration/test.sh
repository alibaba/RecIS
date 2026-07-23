set -ex
#test
TEST_FILES=$(find column-io/test -name "*_test.py")
echo "$(which python)"
for TEST_FILE in ${TEST_FILES}
do
	python -u ${TEST_FILE}
done
