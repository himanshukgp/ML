#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace std;

int main(){
	vector<vector<int> > nums;
	vector<int> temp;
	vector<int> hypothesis;
	vector<int> res;
	int k, count;
	fstream fin;

	fin.open("data1.csv", ios::in); //Open file for reading

	for(size_t i=0; i<20; i++){
		temp.clear();
		for(size_t j = 0; j<9;j++){
			fin >> k; 
			temp.push_back(k);
		}
		nums.push_back(temp);
	}


	hypothesis = nums[0];
	hypothesis[8]=-1;

	for(size_t i=1; i<20; i++){
		if(nums[i][8]==1)
			for(size_t j = 0; j<8;j++)
				if(hypothesis[j]!=-1 && hypothesis[j]!=nums[i][j]) hypothesis[j]=-1;
		
	}

	for(size_t i=0;i<hypothesis.size();i++){
		if(hypothesis[i]==0) {res.push_back(-1*(i+1)); count++;}
		else {res.push_back(i+1); count++;}
	}

	//cout << count;
	//for(size_t i=0; i<res.size(); i++) cout << " " << res[i];

	fin.close();
	return 0;
}