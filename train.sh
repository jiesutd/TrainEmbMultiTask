sh cleanall.sh
cmake .
make JointTrainemb
./JointTrainemb -l -atrain ../debugdata/seg.train.debug -adev ../debugdata/seg.test.debug \
 -btrain ../debugdata/seg.train.debug -bdev ../debugdata/seg.test.debug \
 -ctrain ../debugdata/seg.train.debug -cdev ../debugdata/seg.test.debug \
 -etrain ../debugdata/seg.train.debug -edev ../debugdata/seg.test.debug \
 -char ../debugdata/pmodel.pchar -bichar ../debugdata/pmodel.pbichar -option ../data/option.seg.train
