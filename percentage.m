function [result]=percentage(testGroup,class)
%���������Ե�׼ȷ��,���ྫ��
%��ȷ�ķ������ΪtestGroup��ʵ����Ի�õķ������Ϊclass
count=0;
for i=(1:length(testGroup))
    if class(i)==testGroup(i)
        count=count+1;
    end
end
% fprintf('���ྫ��Ϊ��%f\n' ,count/length(testGroup));
result=count/length(testGroup);
end