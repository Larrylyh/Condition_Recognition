function [result]=percentage(testGroup,class)
%计算分类测试的准确率,分类精度
%正确的分类情况为testGroup，实验测试获得的分类情况为class
count=0;
for i=(1:length(testGroup))
    if class(i)==testGroup(i)
        count=count+1;
    end
end
% fprintf('分类精度为：%f\n' ,count/length(testGroup));
result=count/length(testGroup);
end