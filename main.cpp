#pragma GCC optimize("Ofast","unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
// #pragma GCC target("rdrnd", "popcnt", "avx", "bmi2")

#include <algorithm>
#include <string>
#include <vector>
#include <cstdint>
#include <immintrin.h> //SSE Extensions
#include <bits/stdc++.h> //All main STD libraries
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <atomic>
#include <unordered_set>

using namespace std;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Global Variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~
string fileModel1 = "validate.w32";
uint32_t PARAMS_PER_CELL = 3;                       // Cell is: X - O - Empty
uint32_t INPUT_SIZE = 9*PARAMS_PER_CELL;
vector<uint32_t> TOPOLOGY = {INPUT_SIZE, 3, 1};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Config file with parameters ~~~~~~~~~~~~~~~~~
enum mcts_mode { selfplay, pit, submit };
struct MCTS_Conf {
	//All these initial values will be replaced with the parameters from 
	float cpuct_base = 1.0f;
	float cpuct_inc = 0.0f;
	float cpuct_limit = 1.0f;

	float dirichlet_noise_epsilon = 0.0f;
	float dirichlet_noise_alpha = 1.0f;	// Dirichlet alpha = 10 / n --> Max expected moves
	float dirichlet_decay = 0.00f;
	int num_iters_per_turn = 800;
	float simpleRandomRange = 0.00f;
	bool useTimer = false;
	bool useHeuristicNN = false;
	float PROPAGATE_BASE = 0.7f; //Propagate "WIN/LOSS" with 70% at start
	float PROPAGATE_INC = 0.3f; //Linearly increase +30% until endgame
	int POLICY_BACKP_FIRST = 10; //Similarly , but with percentage of turns, first 30% of turns doesn't have any "temperature",
	int POLICY_BACKP_LAST = 10; //from 30% to (100-10=90%) I linearly sharpen policy to get only the best move, 


	mcts_mode mode;
	MCTS_Conf() {}
	MCTS_Conf(float _cpuct_base, float _cpuct_inc, float _cpuct_limit, float _dirichlet_noise_epsilon,
		float _dirichlet_noise_alpha, float _dirichlet_decay, bool _useTimer, int _num_iters_per_turn,
		float _simpleRandomRange, float _PROPAGATE_BASE, float _PROPAGATE_INC, int _POLICY_BACKP_FIRST, int _POLICY_BACKP_LAST, mcts_mode _mode) {
		cpuct_base = _cpuct_base;
		cpuct_inc = _cpuct_inc;
		cpuct_limit = _cpuct_limit;
		dirichlet_noise_epsilon = _dirichlet_noise_epsilon;
		dirichlet_noise_alpha = _dirichlet_noise_alpha;
		dirichlet_decay = _dirichlet_decay;
		useTimer = _useTimer;
		num_iters_per_turn = _num_iters_per_turn;
		mode = _mode;
		simpleRandomRange = _simpleRandomRange;
		PROPAGATE_BASE = _PROPAGATE_BASE;
		PROPAGATE_INC = _PROPAGATE_INC;
		POLICY_BACKP_FIRST = _POLICY_BACKP_FIRST;
		POLICY_BACKP_LAST = _POLICY_BACKP_LAST;
	}

	string print() {
		string otp = "Conf:";
		otp += " cpuct_base:" + to_string(cpuct_base);
		otp += " CPUCT_inc:" + to_string(cpuct_inc);
		otp += " cpuct_limit:" + to_string(cpuct_limit);
		otp += " DN_e:" + to_string(dirichlet_noise_epsilon);
		otp += " DN_A:" + to_string(dirichlet_noise_alpha);
		otp += " DN_d:" + to_string(dirichlet_decay);
		otp += " Iters:" + to_string(num_iters_per_turn);
		otp += " Mode:" + to_string(mode);
		otp += " Rnd:" + to_string(simpleRandomRange);
		otp += " PROPAGATE_BASE:" + to_string(PROPAGATE_BASE);
		otp += " PROPAGATE_INC:" + to_string(PROPAGATE_INC);
		otp += " POLICY_BACKP_FIRST:" + to_string(POLICY_BACKP_FIRST);
		otp += " POLICY_BACKP_LAST:" + to_string(POLICY_BACKP_LAST);
		return otp;
	}
} default_conf;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NN Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vector<vector<float>> inputWeights = {{0.13488,-0.328414,-0.285128,-0.30394,-0.315525,-0.33117,0.440792,0.245467,0.431413,-0.0867954,0.208683,-0.253549,0.367544,-0.397672,0.402332,-0.275059,-0.378912,-0.250924,0.0629764,-0.412436,-0.315096,-0.147704,0.0570598,0.297492,0.0268105,0.291443,-0.171254,0.393629,-0.431663,-0.0556662,0.0943832,-0.281695,-0.0411239,0.320893,-0.340822,-0.347838,0.275857,0.261584,-0.35467,-0.2063,-0.374651,-0.242986,-0.228155,0.179583,0.283547,0.113766,0.324135,0.323313,0.240136,-0.355949,-0.416029,-0.0662754,0.256964,-0.311379,0.418676,-0.388216,0.116624,0.416769,-0.232403,-0.36017,0.166348,0.368341,0.326348,-0.398484,0.117622,-0.0252647,-0.290102,-0.0886686,0.269013,-0.111395,0.253205,-0.187899,-0.0274297,0.385037,0.205685,-0.231366,0.181787,-0.0414822,0.0517874,-0.355026,0.0982133},
{2.17618,-1.13738,0.983595}};
vector<vector<float>> inputBias = {{-0.00050308,-0.000222007,0.000441785},
{1.34397}};

class SimpleNN{
    public:
        vector<uint32_t> topology;
        vector<vector<float>> values;   // Values of neurons
        vector<vector<float>> coeffs;   // Input weights of each neuron 
        vector<vector<float>> biases;           // Biases
    public:
        SimpleNN(vector<uint32_t> topology, const std::string& f){
            this->topology = topology;
            this->loadWeights(f);
            for(uint32_t i = 0; i < topology.size(); i++){
                values.push_back(vector<float>(topology[i]));
            }
        }
        SimpleNN(vector<uint32_t> topology){
            this->topology = topology;
            coeffs = inputWeights;
            biases = inputBias;
            for(uint32_t i = 0; i < topology.size(); i++){
                values.push_back(vector<float>(topology[i]));
            }
        }
        void dot_product(const uint32_t &layer){
            for(uint32_t i = 0; i < topology[layer]; i++){
                values[layer][i] = 0;
                for(uint32_t j = 0; j < topology[layer-1]; j++){
                    values[layer][i] += values[layer-1][j] * coeffs[layer-1][topology[layer]*j+i];
                }
                values[layer][i] = tanh(values[layer][i] + biases[layer-1][i]);
            }
        }
        void propagate(){
            for(uint32_t i = 1; i < topology.size(); i++){
                dot_product(i);
            }
        }
        float getresults(){
            return values[topology.size()-1][0];
        }
        void loadInput(vector<float> &input){
            uint32_t ctr = 0;
            values[0].assign(input.begin(), input.end());
            if(input.size() != this->values[0].size())cerr<<"Wrong Test size";
        }
        void load(std::ifstream& is, uint32_t &layer){
            uint32_t size = topology[layer-1] * topology[layer];
            float feats[size]; 
		    is.read((char*)feats, size * sizeof(float));
            vector<float> tmp;
            tmp.assign(feats, feats + size);
            this->coeffs.push_back(tmp);
            float bias[topology[layer]];
		    is.read((char*)bias, topology[layer] * sizeof(float));
            vector<float> tmp1;
            tmp1.assign(bias, bias + topology[layer]);
            biases.push_back(tmp1);
            //cout<<"~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            //for(float x : coeffs[coeffs.size()-1])cout<<x<<endl;
            //cout<<"~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
            //for(float x : biases[biases.size()-1])cout<<x<<endl;
            //cout<<"~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        }
        void loadWeights(const std::string& f) {
            std::ifstream is(f, std::ifstream::in | std::ios::binary);
            if (is.good()){
                for (uint32_t i = 1; i < topology.size(); i++) {
                    this->load(is, i);
                }
            }
            else{
                cerr<<"Error #001: Input Error"<<endl;
            }
        }
        void printWeights(std::ofstream& myFile){
            myFile<<"inputWeights = {";
            for(int i = 0; i < coeffs.size(); i++){
                if(i>0) myFile<<",\n";
                myFile<<"{";
                for(int j = 0; j < coeffs[i].size(); j++){
                    if(j>0)myFile<<",";
                    myFile<<coeffs[i][j];
                }
                myFile<<"}";
            }
            myFile<<"}\n";

            myFile<<"inputBias = {";
            for(int i = 0; i < biases.size(); i++){
                if(i>0) myFile<<",\n";
                myFile<<"{";
                for(int j = 0; j < biases[i].size(); j++){
                    if(j>0)myFile<<",";
                    myFile<<biases[i][j];
                }
                myFile<<"}";
            }
            myFile<<"}";
        }
};

void writeFile(SimpleNN nn){
    ofstream myfile ("example.txt");
    if (myfile.is_open())
    {
        auto begin = myfile.tellp();
        nn.printWeights(myfile);
        auto end = myfile.tellp();
        cout<<"File written. File size: "<<end - begin<< " bytes. \n";
        myfile.close();
    }
    else cout << "Unable to open file";
}

void ValidateModel(string file, string str_inputs) {
    SimpleNN testNet = SimpleNN(TOPOLOGY , file);

	vector<float> values;
	istringstream iss(str_inputs);
	copy(istream_iterator<float>(iss), istream_iterator<float>(), back_inserter(values));
    testNet.loadInput(values);
	cerr << "Inputs:" << str_inputs << endl;
	float val = 0.0f;

	testNet.propagate();

	//Set that value as Score;
	val = testNet.getresults();

	//policy->setElement(6, -99999999.99f);
	//policy->setElement(7, -99999999.99f);
	//Activation_Softmax(*policy, *policy);
	cerr << "VALIDATE MODEL:" << endl;
	cerr << "Value Expected :[" << val << "]" << endl;
	cerr << endl;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MCTS Algorithm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Node of MCTS_Tree
struct Node{
    float nnEval;
    float timesVisited;
    uint32_t firstChild;
    uint32_t parentAddr;
    uint16_t gridCell;   //Gridcell where move will be placed
    uint8_t childNr;
    bool solved;
};
float C = 0.5;   //Represents the C in the MCTS ucb equation:
                 //value + C * sqrt(log(parent visits) / visits)
float FTP = 1.5; //If the Node hasn't been visited it's value is calculated as:
                 //value + FTP


class MCTS_Tree{
    public:
        Node* nodes = (Node*) malloc(10000000 * sizeof(Node));   // Node Pool
        SimpleNN net;
        vector<float> gameState;
        uint32_t rootNodeAddr;
        uint32_t lastNodeAddr;
        MCTS_Conf myConf;
    public:
        MCTS_Tree(MCTS_Conf conf, vector<uint32_t> topology, const string &str): 
            net(topology, str),
            gameState(topology[0])
        {
            for(int i = 0; i < topology[0]; i++){
                if(i%PARAMS_PER_CELL == 2) gameState[i] = 1;
            }
            myConf = conf;
            rootNodeAddr = 0;
            lastNodeAddr = 0;
        }
        MCTS_Tree(MCTS_Conf conf, vector<uint32_t> topology): 
            net(topology),
            gameState(topology[0])
        {
            for(int i = 0; i < topology[0]; i++){
                if(i%PARAMS_PER_CELL == 2) gameState[i] = 1;
            }
            myConf = conf;
            rootNodeAddr = 0;
            lastNodeAddr = 0;
        }
        uint32_t newNode(const uint32_t &parentAddr, const uint16_t &gridCell){
            nodes[lastNodeAddr].gridCell = gridCell;
            nodes[lastNodeAddr].parentAddr = parentAddr;
            lastNodeAddr++;
            return lastNodeAddr-1;
        }

        uint32_t findBestChild(const uint32_t &myNodeAddr){
            nodes[myNodeAddr].timesVisited++;
            uint32_t bestChild = 0;
            double logParent = C * logf(nodes[myNodeAddr].timesVisited), maxVal = -2;
            for(uint32_t child = nodes[myNodeAddr].firstChild; 
                    child < nodes[myNodeAddr].firstChild + nodes[myNodeAddr].childNr; child++){
                double val = nodes[child].nnEval + 
                    ((nodes[child].timesVisited == 0) ? FTP : (logParent/nodes[child].timesVisited));
                if(maxVal < val){
                    maxVal = val;
                    bestChild = child;
                }
            }
            if(bestChild == 0) cerr<<"Error #002: No children"<<endl;
            return  bestChild;
        }


        int8_t checkWinning(const bool &oturn){
            vector<bool> t(9);      
            bool checker = false;   // To check if gridCell is Tied
            uint8_t ctr = 0;
            // fill t vector
            for(uint32_t i = 0; i < 9*PARAMS_PER_CELL; i += PARAMS_PER_CELL){
                t[ctr++] = gameState[i+oturn];
                checker |= (gameState[i+2]==true);
            }
            // check if winning
            bool flag = false;
            if(t[0]==true && t[1]==true && t[2]==true)flag = true;
            if(t[3]==true && t[4]==true && t[5]==true)flag = true;
            if(t[6]==true && t[7]==true && t[8]==true)flag = true;
            if(t[0]==true && t[3]==true && t[6]==true)flag = true;
            if(t[1]==true && t[4]==true && t[7]==true)flag = true;
            if(t[2]==true && t[5]==true && t[8]==true)flag = true;
            if(t[0]==true && t[4]==true && t[8]==true)flag = true;
            if(t[6]==true && t[4]==true && t[2]==true)flag = true;
            if(flag == true){
                return oturn ? 2 : 4;
            }
            // If cells are full: draw
            if(checker == false) {
                return 3;
            }
            return 0;
        }

        void playMove(const uint32_t &gridCell, const bool &oturn){
            // Quick check if the gamestate is used
            if(gameState[gridCell] || gameState[gridCell+1] || !gameState[gridCell+2]){
                cerr<<"Error #004: Some wrong Parameter! "<<gameState[gridCell] << gameState[gridCell+1] << !gameState[gridCell+2]<<endl;
                exit(1);
            }
            gameState[gridCell + oturn] = true;
            gameState[gridCell + 2] = false;
        }

        vector<uint32_t> findNextMoves(){
            vector<uint32_t> res;
            for(uint32_t i = 0; i < 9; i++){
                if(gameState[i*PARAMS_PER_CELL + 2] == true) 
                    res.push_back(i*PARAMS_PER_CELL);
            }
            if(res.size() != 0) return res;
            if(res.size() == 0){
                printState();
                cerr<<"Error #005: No moves left"<<endl;
                exit(1);
            }
            return res;
        }
        
        void expandOnce(uint32_t myNode, bool oturn){
            // Traverse down the tree
            while(nodes[myNode].timesVisited!=0){
                myNode = findBestChild(myNode);
                if(nodes[myNode].solved == true){
                    nodes[myNode].timesVisited += 1;
                    return;
                }
                playMove(nodes[myNode].gridCell, oturn);
                oturn = !oturn;
            }
            nodes[myNode].timesVisited = 1;
            // Get next moves
            vector<uint32_t> nextMoves = findNextMoves();
            // Store grid for later
            vector<bool> copyGrid(INPUT_SIZE);
            copyGrid.assign(gameState.begin(), gameState.end());
            // Init some parameters
            nodes[myNode].firstChild = lastNodeAddr;
            nodes[myNode].childNr = nextMoves.size();
            // Create child, play move, evaluate
            uint32_t currAddr;
            for(uint32_t x : nextMoves){
                // Reset Values
                gameState.assign(copyGrid.begin(), copyGrid.end());
                //Create Node, play move
                currAddr = newNode(myNode, x);
                playMove(x, oturn);
                //Check if gridcell is winning
                int8_t r = checkWinning(oturn);
                nodes[currAddr].solved = (r>1);
                if(r>1){
                    float f = r-3; // 1,0,-1 if X is winning,tie,losing
                    nodes[currAddr].nnEval = oturn ? -f : f;
                    nodes[myNode].nnEval = -max(-nodes[myNode].nnEval, nodes[currAddr].nnEval);
                    continue;
                }
                net.loadInput(gameState);
                net.propagate();
                nodes[currAddr].nnEval = oturn ? -net.getresults() : net.getresults();
                nodes[myNode].nnEval = -max(-nodes[myNode].nnEval, nodes[currAddr].nnEval);
            }
            // Minimax up the tree until no changes
            // while(nodes[myNode].nnEval > -nodes[nodes[myNode].parentAddr].nnEval){
            //     nodes[nodes[myNode].parentAddr].nnEval = -nodes[myNode].nnEval;
            //     myNode = nodes[myNode].parentAddr;
            // }
            while(nodes[myNode].parentAddr != myNode){
                myNode = nodes[myNode].parentAddr;
                float bestval = -2;
                for(uint32_t child = nodes[myNode].firstChild; child < nodes[myNode].firstChild + nodes[myNode].childNr; child++){
                    bestval = max(bestval, nodes[child].nnEval); 
                }
                if (bestval == nodes[myNode].nnEval) break;
                nodes[myNode].nnEval = -bestval;
            }
        }

        // Select child with most Visits
        uint32_t selectMove(){
            uint32_t bestTV = 0, bestChild = 0;
            for(uint32_t i = nodes[rootNodeAddr].firstChild; i < nodes[rootNodeAddr].firstChild+nodes[rootNodeAddr].childNr; i++){
                if(nodes[i].timesVisited >= bestTV){
                    bestChild = i;
                    bestTV = nodes[i].timesVisited;
                }
            }
            // So that backpropagation stops
            nodes[bestChild].parentAddr = bestChild;
            rootNodeAddr = bestChild;
            return nodes[bestChild].gridCell;
        }

        pair<uint8_t, uint8_t> playRound(const int32_t &x, const int32_t &y){
            float timestamp = 0.1;
            if(lastNodeAddr != 0){
                uint32_t gridCell = ((x%3)*3+y%3)*PARAMS_PER_CELL;
                int32_t moveChild = -1;
                for(int i = nodes[rootNodeAddr].firstChild; i<nodes[rootNodeAddr].firstChild+nodes[rootNodeAddr].childNr; i++){
                    if(nodes[i].gridCell == gridCell){
                        moveChild = i;
                        break;
                    }
                }
                if(moveChild==-1){
                    cerr<<"Error #006: Child doesn't exist"<<endl;
                    exit(1);
                }
                playMove(gridCell, true);
                rootNodeAddr = moveChild;
                nodes[moveChild].parentAddr = moveChild;
            } 
            else {
                timestamp = 1.0;
                if(x!=-1){
                    uint32_t gridCell = ((x%3)*3+y%3)*PARAMS_PER_CELL;
                    newNode(0, gridCell);
                    playMove(gridCell, true);
                } else {
                    newNode(0, (4*9 + 4)*PARAMS_PER_CELL);
                }
            }
            if(nodes[rootNodeAddr].solved){
                cout<<endl<<"GG!"<<endl;
                exit(0);
            }
            vector<float> gridCopy(INPUT_SIZE);
            gridCopy.assign(gameState.begin(), gameState.end());
            auto start = std::chrono::high_resolution_clock::now();
            int d = lastNodeAddr;
            // while(chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - start).count() < timestamp){
            for(int i = 0; i < 100; i++){
                expandOnce(rootNodeAddr, false);
                gameState.assign(gridCopy.begin(), gridCopy.end());
            }
            printState();
            uint32_t bestMove = selectMove()/PARAMS_PER_CELL;
            playMove(bestMove*PARAMS_PER_CELL, false);
            uint8_t bgc = bestMove/9, sgc = bestMove%9;
            return pair<uint8_t, uint8_t>((bgc/3)*3+(sgc/3), (bgc%3)*3+(sgc%3));
            // cout<<unsigned(move.first)<<" "<<unsigned(move.second)<<" ,"<<getEval()<<getChildEval()<<endl;
            // if(nodes[rootNodeAddr].solved){
            //     cout<<"GG!"<<endl;
            //     exit(0);
            // }
            // return move;
        }

        void printChildren(int NodeAddr){
            for(uint32_t i = nodes[NodeAddr].firstChild; i < nodes[NodeAddr].firstChild + nodes[NodeAddr].childNr; i++){
                cerr<<"("<<nodes[i].gridCell/PARAMS_PER_CELL<<","<< nodes[i].nnEval<<","<< nodes[i].timesVisited<<")";
            }
            cerr<<endl;
        }
        
        float findChildEval(int NodeAddr, int gridcell){
            for(uint32_t i = nodes[NodeAddr].firstChild; i < nodes[NodeAddr].firstChild + nodes[NodeAddr].childNr; i++){
                if(nodes[i].gridCell/PARAMS_PER_CELL == gridcell) return nodes[i].nnEval;
            }
            cerr<<"No such child"<<endl;
            return 0;
        }

        void printState(){
            for(int i = 0; i < gameState.size(); i+=PARAMS_PER_CELL){
                if(i%(PARAMS_PER_CELL*3)==0) {
                    if(i) cout<<endl<<"|           |           |           |"<<endl;
                    cout<<"+-----------+-----------+-----------+"<<endl;
                    if(i < gameState.size() - PARAMS_PER_CELL) cout<<"|           |           |           |"<<endl<<"|";
                }
                if(gameState[i]==true)cout<<"     X     ";
                if(gameState[i+1]==true)cout<<"     O     ";
                if(gameState[i+2]==true){
                    if(nodes[rootNodeAddr].solved) cout<<"     -     ";
                    else cout<<setw(11) << findChildEval(rootNodeAddr, i/PARAMS_PER_CELL);
                }
                cout<<"|";
            }
            cout<<endl<<"|           |           |           |"<<endl<<"+-----------+-----------+-----------+"<<endl;
        }
        bool won(){
            return nodes[rootNodeAddr].solved;
        }
        float getEval(){
            return nodes[rootNodeAddr].nnEval;
        }
        int getChildEval(){
            if(nodes[rootNodeAddr].childNr==0)return 2;
            cerr<<"(";
            for(int i = nodes[rootNodeAddr].firstChild; i < nodes[rootNodeAddr].firstChild + nodes[rootNodeAddr].childNr; i++){
                cerr<<nodes[i].nnEval<<"_";
            }
            cerr<<")";
            return 6;
        }
};


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Play Game ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct Game{
    public:
        vector<vector<int>> gameState;
    public:
        Game():
            gameState(9,vector<int>(9))
            {
            }
        void reset(){
            for(int i = 0; i < 9; i++){
                for(int j = 0; j < 9; j++){
                    gameState[i][j] = 0;
                }
            }
        }

        void fillGrid(int x, int y, bool oturn){
            for(int i = (x/3)*3; i < (x/3)*3+3; i++){
                for(int j = (y/3)*3; j < (y/3)*3+3; j++){
                    gameState[i][j] = oturn?-1:1;
                }
            }
        }


        void playMove(uint32_t x, uint32_t y, bool oturn){
            if(gameState[x][y] != 0)
            {
                cerr<<"Error #007: Invalid Move"<<endl;
                exit(1);
            }
            gameState[x][y] = oturn ? -1 : 1;
            // checkWinningGC(x, y, oturn);
        }
        bool checkDone(){
            for(int i = 0; i < 9; i++){
                for(int j = 0; j < 9; j++){
                   if(gameState[i][j]==0) 
                       return false; 
                }
            }
            return true;
        }
        void printGrid(){
            int ctr=0;
            char arr[3] = {
                'O', '-', 'X'
            };
            for(vector<int> x : gameState){
                for(int y : x){
                    cerr<<arr[y+1]<<",";
                    if((++ctr)%3==0) cout<<"|";
                }
                cout<<endl;
            }
        }
};

//*********************************************** Sampling Stuff ***************************************************************/
struct SampleInfo {
	vector<float> I;
    float P;
	int N;
	int win, draw, loss;
};
struct SamplesFile {
	string file;
	unordered_map<size_t, SampleInfo> samples;
	SamplesFile(string _file) {
		file = _file;
	}
};
vector<SamplesFile> sampleFiles;
std::mutex mutex_selfGames;

//Find samplesFile that contains position
SamplesFile* getSampleFile(string file) {
	SamplesFile* sFile = nullptr;
	for (auto&s : sampleFiles) {
		if (s.file == file) {
			sFile = &s;
			break;
		}
	}
	if (sFile == nullptr) {
		SamplesFile newFile(file);
		sampleFiles.emplace_back(newFile);
		sFile = &sampleFiles.back();
	}
	return sFile;
}

template <class T>
inline void hash_combine(std::size_t& seed, T const& v)
{
	seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template<typename T>
size_t hashVector(vector<T>& in) {
	size_t size = in.size();
	size_t seed = 0;
	for (size_t i = 0; i < size; i++)
		//Combine the hash of the current vector with the hashes of the previous ones
		hash_combine(seed, in[i]);
	return seed;
}
void insertNewSample(SamplesFile* sFile, SampleInfo& S) {
	size_t HASH = hashVector(S.I);
	auto hasSample = sFile->samples.find(HASH);
	if (hasSample == sFile->samples.end()) //NEW
	{
		sFile->samples.emplace(HASH, S);
	}
	else {
		hasSample->second.N += S.N;
		hasSample->second.P += S.P;
		hasSample->second.win += S.win;
		hasSample->second.loss += S.loss;
		hasSample->second.draw += S.draw;
	}
}


int processSamplesFile(string file, const int INPUT_SIZE, const int OUTPUT_SIZE) {
	auto t0 = chrono::high_resolution_clock::now() ;
	cerr << "Processing " << file;
	//Inputs + POLICY + VALUE
	mutex_selfGames.lock();
	SamplesFile* sFile = getSampleFile(file);
	ifstream F(file, std::ios::in | std::ios::binary);
	if (!F.good()) {
		mutex_selfGames.unlock();
		cerr << "Error reading file:" << file << endl;
		return true;
	}
	//Create space
	string line;
	SampleInfo S;
	S.I.resize(INPUT_SIZE);
	// S.P.resize(OUTPUT_SIZE);
	int linesProcessed = 0;
	F.seekg(0);
	int PROCESSED_SAMPLES = 0;

	while (!F.eof())// (getline(F, line))
	{
		++PROCESSED_SAMPLES;
		++linesProcessed;
		S.N = 1;
		F.read(reinterpret_cast<char*>(&S.I[0]), INPUT_SIZE * sizeof(float));
		if (F.eof())
			break;
		F.read(reinterpret_cast<char*>(&S.P), OUTPUT_SIZE * sizeof(float));
		float fN = (float)S.N;
		F.read(reinterpret_cast<char*>(&fN), sizeof(fN));
		S.win = 0;
		S.loss = 0;
		S.draw = 0;
		if (S.P > 0.45f)
		{
			++S.win;
		}
		else if (S.P < -0.45f)
		{
			++S.loss;
		}
		else ++S.draw;
		insertNewSample(sFile, S);
	}
	mutex_selfGames.unlock();
	F.close();
	cerr << " Done. Lines:" << linesProcessed << " PROCESSED:" << PROCESSED_SAMPLES 
        << " Unique:" << sFile->samples.size() << " T:" << chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - t0).count() << endl;
	return true;
}

bool saveSamplesFile(string file) {
	SamplesFile* sFile = getSampleFile(file);
	if (sFile == nullptr)
	{
		cerr << "Error, samples for  " << file << " not found!" << endl;
		return false;
	}
	ofstream outputfile(file, std::ios::out | std::ios::binary);
	if (!outputfile.good())
	{
		cerr << "Invalid output file:" << file << endl;
		return false;
	}

	for (auto& ttt : sFile->samples)
	{
		SampleInfo& s = ttt.second;
		outputfile.write(reinterpret_cast<char*>(&s.I[0]), (int)s.I.size() * sizeof(float));
		//Averaging
		if (s.N > 1)
		{
			float divide = 1.0f / (float)s.N;
			s.P *= divide;
		}
		outputfile.write(reinterpret_cast<char*>(&s.P),  sizeof(float));
		float fN = (float)s.N;
		outputfile.write(reinterpret_cast<char*>(&fN), (int) sizeof(fN));
	}
	outputfile.close();
	return true;
}

//*********************************************** PIT WORKER - Match 2 models and get a winrate***************************************************************/
atomic<int> Pit_V1;
atomic<int> Pit_V2;
atomic<int> Pit_Draw;
atomic<int> matches;
//One worker per thread. Uses <atomic> to avoid race conditions
int WorkerOneRun(int ID, string fileModel1, string fileModel2, MCTS_Conf conf1, MCTS_Conf conf2){
    MCTS_Tree player1 = MCTS_Tree(conf1, TOPOLOGY, fileModel1);
    MCTS_Tree player2 = MCTS_Tree(conf2, TOPOLOGY, fileModel2);
    Game game = Game();
    pair<int8_t, int8_t> move = pair<int,int>(-1,-1);
    bool oturn = false;
    int x=-1,y=-1;
    bool done = false;
    int res = -2;
    while(!game.checkDone()){
        if(oturn == false){
            move = player1.playRound(x,y);
            done = player1.won();
            if(done){
                res = player1.getEval();
            }
        }
        else{
            move = player2.playRound(x,y);
            done = player2.won();
            if(done){
                res = -player2.getEval();
            }
        }
        x = move.first;
        y = move.second;
        //cout<<"Move:"<<x<<","<<y<<endl;
        game.playMove(x, y, oturn);
        //game.printGrid();
        if(done){
            //cerr<<"Output:"<<player1.getEval()<<","<<player1.getChildEval()<<"|" << player2.getEval()<<","<<player2.getChildEval()<<endl;
            break;
        }
        oturn = !oturn;
    }
    //cerr<<res<<endl;
    delete player1.nodes; 
    delete player2.nodes;
    return res;
}

void Worker_Pit(int ID, string fileModel1, string fileModel2, int matchperWorker, MCTS_Conf conf1, MCTS_Conf conf2){
	cerr << "Worker " << ID << " will play " << matchperWorker << " matches" << endl;
    int output;
    for(int i = 0; i < matchperWorker; i++){
        if(i&1)
            output = WorkerOneRun(ID, fileModel1, fileModel2, conf1, conf2);
        else
            output = -WorkerOneRun(ID, fileModel2, fileModel1, conf2, conf1);
        if(output==1){
            ++Pit_V1;
        } else if(output == -1){
            ++Pit_V2;
        } else {
            ++Pit_Draw;
        }
        ++matches;
		{ //notify
			int totalGames = Pit_V1 + Pit_V2 + Pit_Draw;
			float winrate = 100.0f * ((float)Pit_V1 + 0.5f * (float)Pit_Draw) / (float)(totalGames);
			cerr << "Worker " << ID << ": " << Pit_V1 << "/" << Pit_V2 << "/" << Pit_Draw << ":" << winrate << "%";
			cerr<< endl;
		}
    }
}

MCTS_Conf selfPlay_Mode(1.0f, 0.0f, 1.0f, 0.25f, 1.0f, 0.01f, false, 1200, 0.00f, 0.7f, 0.3f, 10, 10, mcts_mode::selfplay);

int max(const int &a, const int &b){
    return a > b ? a : b;
}

// Play games against other bot to determine better bot
int pitPlay(int argc, char* argv[]){
    Pit_V1 = 0;
	Pit_V2 = 0;
	Pit_Draw = 0;
	matches = 0;
	uint32_t agc = 2;
	uint32_t THREADS = atoi(argv[agc++]);
	uint32_t matchCount = atoi(argv[agc++]);
	uint32_t matchperWorker = matchCount / THREADS;

	string fileModel1 = string(argv[agc++]);
	MCTS_Conf conf1 = selfPlay_Mode;
	if (argc > agc) conf1.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf1.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf1.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_LAST = atoi(argv[agc++]);

	string fileModel2 = string(argv[agc++]);
	MCTS_Conf conf2 = conf1;
	if (argc > agc) conf2.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf2.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf2.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_LAST = atoi(argv[agc++]);

	cerr << "pitplay m1:" << fileModel1 << " m2:" << fileModel2 << " Th:" << THREADS << " N:" << matchCount << " C1:" << conf1.print() << " C2:" << conf2.print() << endl;
	vector<thread> threads(max(1, THREADS));
    for (int i = 0; i < max(1, THREADS); i++){
		threads[i] = thread(Worker_Pit, i, fileModel1 + ".w32", fileModel2 + ".w32", matchperWorker, conf1, conf2);
	}
	for (int i = 0; i < max(1, THREADS); i++){
		threads[i].join();
	}
	int totalGames = Pit_V1 + Pit_V2 + Pit_Draw;
	float winrate = 100.0f * ((float)Pit_V1 + 0.5f * (float)Pit_Draw) / (float)(totalGames);
	string PitFile = "./pitresults/Pit_" + fileModel1 + "_" + fileModel2 + "_" + to_string(winrate) + ".txt";
	ofstream f(PitFile);
	if (f.good())
	{
		f << winrate;
		f.close();
	}
	cout << winrate << endl;
	return 0;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Self play and Store games, score to the dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

struct Move {
    vector<float> gamestate;
    float meanScore;
    bool ignoreDontSave;
    float originalValue;
    float backtrackValue;
    Move(vector<float> gameState, bool ignoreDontSave){
        this->ignoreDontSave = ignoreDontSave;
        gamestate.assign(gameState.begin(), gameState.end());
    }
};
struct ReplayGame {
	vector<Move> moves;
    vector<float> meanScore;
	float reward; // -1.0 to 1.0
};
struct ReplayBuffer {
	vector<ReplayGame> games;
};

ReplayBuffer selfGames;
Move getReplayBuffer(vector<float> gameState, float nnEval){
    Move m = Move(gameState, false);
    // Need to calculate float mean.
    m.meanScore = nnEval;
    return m;
}
void swap(vector<float> &state){
    int x = 0;
    for(int i = 0; i < state.size(); i+=PARAMS_PER_CELL){
        x = state[i];
        state[i] = state[i+1];
        state[i+1] = x;
    }
}
//One worker per thread. Uses <atomic> to avoid race conditions
int self_WorkerOneRun(int ID, string fileModel1, string fileModel2, MCTS_Conf conf1, MCTS_Conf conf2){
    MCTS_Tree player1 = MCTS_Tree(conf1, TOPOLOGY, fileModel1);
    MCTS_Tree player2 = MCTS_Tree(conf2, TOPOLOGY, fileModel2);
    Game game = Game();
    pair<int8_t, int8_t> move = pair<int,int>(-1,-1);
    bool oturn = false;
    int x=-1,y=-1;
    bool done = false;
    int res = -2;
    float eval = 0;
    ReplayGame RG;
    vector<float> gState(TOPOLOGY[0]);
    while(!game.checkDone()){
        if(oturn == false){
            move = player1.playRound(x,y);
            done = player1.won();
            if(done){
                res = player1.getEval();
            }
            eval = player1.getEval();
            gState.assign(player1.gameState.begin(), player1.gameState.end());
        }
        else{
            move = player2.playRound(x,y);
            done = player2.won();
            if(done){
                res = -player2.getEval();
            }
            eval = player2.getEval();
            gState.assign(player2.gameState.begin(), player2.gameState.end());
            swap(gState);
        }
        x = move.first;
        y = move.second;
        //cout<<"Move:"<<x<<","<<y<<endl;
        game.playMove(x, y, oturn);
        //game.printGrid();
        if(done){
            //cerr<<"Output:"<<player1.getEval()<<","<<player1.getChildEval()<<"|" << player2.getEval()<<","<<player2.getChildEval()<<endl;
            break;
        }
        RG.moves.emplace_back(getReplayBuffer(gState, eval));
        oturn = !oturn;
    }
    RG.reward = res;
    cerr << "Worker " << ID << ":" << matches << " W:" << RG.reward << " " << res <<  " T:" << oturn<< " moves:" << RG.moves.size() <<"\n";
    int totalMovesInReplay = RG.moves.size();
    //Backpropagate reward
    int accMoves = 0;
    float sReward = RG.reward;
    for (auto& r : RG.moves) {
        float rewardFactor = conf1.PROPAGATE_BASE + conf1.PROPAGATE_INC * ((float)accMoves / (float)totalMovesInReplay);
        r.originalValue = r.meanScore;
        r.backtrackValue = sReward;
        r.meanScore = rewardFactor * sReward + (1.0f - rewardFactor) * r.meanScore;
        sReward = -sReward;
        accMoves += 1;
    }
    //Dump data
    mutex_selfGames.lock();
    selfGames.games.emplace_back(RG);
    mutex_selfGames.unlock();

    delete player1.nodes; 
    delete player2.nodes;
    return res;
}

void Worker_SelfPlay(int ID, string fileModel1, string fileModel2, int matchperWorker, MCTS_Conf conf1, MCTS_Conf conf2){
	cerr << "Worker " << ID << " will play " << matchperWorker << " matches" << "\n";
    int output;
    for(int i = 0; i < matchperWorker; i++){
        if(i&1)
            output = self_WorkerOneRun(ID, fileModel1, fileModel2, conf1, conf2);
        else
            output = -self_WorkerOneRun(ID, fileModel2, fileModel1, conf2, conf1);
        ++matches;
    }
}

//Read inputs, create <THREADS> Self-play workers and then save the winrate on a file
int selfPlay(int argc, char* argv[])
{
	matches = 0;
	int agc = 2;
	//Read command line parameters
	int THREADS = atoi(argv[agc++]);
	int matchCount = atoi(argv[agc++]);
	int matchperWorker = matchCount / THREADS;

	string fileModel1 = string(argv[agc++]);
	MCTS_Conf conf1 = selfPlay_Mode;
	if (argc > agc) conf1.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf1.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf1.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_LAST = atoi(argv[agc++]);

	string fileModel2 = string(argv[agc++]);
	MCTS_Conf conf2 = conf1;
	if (argc > agc) conf2.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf2.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf2.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_LAST = atoi(argv[agc++]);

	//Prepare destination file
	string samplesFile = "./traindata/Replay_" + fileModel1 + "_" + fileModel2 + ".dat";
	//If it already exists, read all samples because it will be updated.
	processSamplesFile(samplesFile, INPUT_SIZE, 1); //value as output
	cerr << "selfplay m1:" << fileModel1 << " m2:" << fileModel2 << " Th:" << THREADS << " N:" << matchCount << " C1:" << conf1.print() << " C2:" << conf2.print() << "\n";
	selfGames.games.resize(0);
	selfGames.games.reserve(matchCount);
    
	//Threading, each worker is independent, but all will add moves to the same 
    //Replay Buffer (using mutex to avoid race conditions)
	vector<thread> threads(max(1, THREADS));
	for (int i = 0; i < max(1, THREADS); i++){
		threads[i] = thread(Worker_SelfPlay, i, fileModel1 + ".w32", fileModel2 + ".w32", matchperWorker, conf1, conf2);
	}

	for (int i = 0; i < max(1, THREADS); i++){
		threads[i].join();
	}

	//The replay buffer must be deduplicated with existing samples. 
    //If a gamestate appears in two samples, it will sum the policy and value, and N will be increased.
	mutex_selfGames.lock();
	SamplesFile* sFile = getSampleFile(samplesFile);
	for (auto&G : selfGames.games){
		//Backpropagated endgame value in turns before an ignoreDontSave might not be correct
		//Maybe a player win because the opponent did a random move on a critical point
		//So it's safer to just ignore those previous samples.
		bool ign = false;
		if (G.moves.size() > 1)
		for (int i=(int)G.moves.size()-1;i>=0;--i){
			if (G.moves[i].ignoreDontSave)
			{
				ign = true;
			}
			else if (ign)
			{
				G.moves[i].ignoreDontSave = true;
			}
		} for (auto& R : G.moves){
			if (R.ignoreDontSave)
				continue;
			//Convert the Replay Buffer to a SampleInfo. I did this way because previously, 
            //I just saved all samples individually. This took too much disk space, was much slower.
			SampleInfo S;
			S.N = 1;
			S.I = R.gamestate;
			S.P = (R.meanScore);
			S.win = 0;
			S.loss = 0;
			S.draw = 0;
			if (S.P > 0.45f)
			{
				++S.win;
			}
			else if (S.P < -0.45f)
			{
				++S.loss;
			}
			else ++S.draw;
			//Insert sample to existing ones, deduplicating them.
			insertNewSample(sFile, S);
		}
	}
	//Store in file
	saveSamplesFile(samplesFile);
	mutex_selfGames.unlock();
	return 0;
}
// Submission code
void CGPlay(MCTS_Conf gameConf, vector<uint32_t> top, string fileModel){
    MCTS_Tree gameTree = MCTS_Tree(gameConf, top);
    // writeFile(gameTree.net);
    int x, y;
    pair<int, int> move;
    int validActionCount, row, col;
    while(true){
        scanf("%i %i",&x, &y);
        // scanf("%i", &validActionCount);
        // for (int i = 0; i < validActionCount; i++) scanf("%i %i", &row, &col);
        move = gameTree.playRound(x,y);
        cout<<unsigned(move.first)<<" "<<unsigned(move.second)<<endl;
        gameTree.printState();
        if(gameTree.won()){
            cout<<"GG!"<<endl;
            exit(0);
        }
    }
}

//Read inputs, create <THREADS> Pit workers and then save the winrate on a file

int main(int argc, char* argv[]) {
    if(argc==1){
        CGPlay(selfPlay_Mode, TOPOLOGY, fileModel1);
    }
    fileModel1 = "validate.w32";
	string tmpOption(argv[1]);
    if (tmpOption == "pitplay"){
        return pitPlay(argc, argv);
    }
    if (tmpOption == "selfplay"){
        return selfPlay(argc, argv);
    }
    Worker_Pit(0, fileModel1, fileModel1, 10, selfPlay_Mode, selfPlay_Mode);
    
    // ValidateModel("validate.w32",	R"(0.5648349443032249 0.4528574847709059 0.8635838522750298 0.8047026304930922 0.2368564224839521 0.2752383230626436 0.2731484989258365 0.625065698455672 0.1516452447982508 0.06898667924462787 0.33778444922725037 0.1039728368839663 0.4239481519779893 0.3790060317541506 0.9539233121241609 0.5962166371807451 0.1471421185674726 0.8335892821461448 0.021085748736165977 0.45073086738441914 0.7453783309153537 0.6891678570662427 0.5246036837840751 0.7326045596824818 0.17858709387740657 0.03500578239560048 0.9225602962200417)");
    //SimpleNN someNN = SimpleNN( TOPOLOGY, "validate.w32");
    //writeFile(someNN);
}

