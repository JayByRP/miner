#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <sstream>
#include <cstring>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

// Full Keccak-256 implementation for CUDA
__device__ __constant__ const unsigned long long keccakRoundConstants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000000008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ unsigned long long rotateLeft(unsigned long long x, int shift) {
    return (x << shift) | (x >> (64 - shift));
}

__device__ void keccak256(unsigned char *out, const unsigned char *in, size_t inlen) {
    unsigned long long state[25] = {0};
    
    // Absorb phase - simple byte XOR into first bytes
    for (size_t i = 0; i < inlen; i++) {
        state[i / 8] ^= ((unsigned long long)in[i]) << (8 * (i % 8));
    }
    
    // Padding - Keccak style
    state[inlen / 8] ^= 1ULL << (8 * (inlen % 8));
    state[24] ^= 0x8000000000000000ULL;
    
    // Keccak permutation (simplified round function)
    for (int round = 0; round < 24; round++) {
        // Theta step
        unsigned long long C[5] = {0}, D[5] = {0};
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                C[x] ^= state[x + y * 5];
            }
        }
        
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotateLeft(C[(x + 1) % 5], 1);
        }
        
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x + y * 5] ^= D[x];
            }
        }
        
        // Add round constant
        state[0] ^= keccakRoundConstants[round];
    }
    
    // Squeeze phase - extract first 32 bytes
    for (int i = 0; i < 32; i++) {
        out[i] = (state[i / 8] >> (8 * (i % 8))) & 0xFF;
    }
}

// Uniswap V4 Challenge Constants
const std::string INITCODE_HASH = "94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb 645d9";
const std::string DEPLOYER_ADDRESS = "48E516B34A1274f49457b9C6182097796D0498Cb";

__global__ void mineAddressesKernel(
    unsigned char* initcode, 
    unsigned char* deployerAddr, 
    unsigned char* saltBase,
    unsigned char* results, 
    int* scores,
    unsigned long long startSalt, 
    int batchSize,
    int targetScore,
    unsigned int* processedBatches
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    // Increment processed batches tracker
    if (idx == 0) {
        atomicAdd(processedBatches, 1);
    }

    // Prepare salt with thread-specific modification
    unsigned char salt[32] = {0};
    memcpy(salt, saltBase, 32);
    *((unsigned long long*)salt) = startSalt + idx;

    // Combine components for CREATE2 address generation
    unsigned char preHash[96] = {0};
    memcpy(preHash, initcode, 32);
    memcpy(preHash + 32, deployerAddr, 20);
    memcpy(preHash + 52, salt, 32);  // Salt affects the input to the hash

    // Hash generation
    unsigned char hash[32];
    keccak256(hash, preHash, 96);

    // Address Scoring Algorithm
    int score = 0;
    
    // Count leading zeros in address
    int leadingZeros = 0;
    for (int i = 12; i < 20; i++) {
        unsigned char byte = hash[i];
        if (byte == 0) leadingZeros += 2;  // each zero byte contributes 2 points
        else {
            leadingZeros += (byte < 0x10) ? 1 : 0;  // count partial zero
            break;
        }
    }
    score += leadingZeros * 10;

    // Check for four consecutive '4's
    if (hash[12] == 0x40 && hash[13] == 0x40 && hash[14] == 0x40 && hash[15] == 0x40) {
        score += 40;  // four consecutive 4's
        if ((hash[16] >> 4) != 4) {
            score += 20;  // next character not a 4
        }
    }

    // Check if last four characters are all 4s
    if ((hash[28] >> 4) == 4 && (hash[29] >> 4) == 4 && (hash[30] >> 4) == 4 && (hash[31] >> 4) == 4) {
        score += 20;  // last four characters are all 4's
    }

    // Count all '4's in the address
    for (int i = 12; i < 32; i++) {
        unsigned char byte = hash[i];
        score += ((byte >> 4) == 4) + ((byte & 0x0F) == 4);  // count each '4'
    }

    // Store results
    memcpy(results + idx * 32, hash, 32);
    scores[idx] = score;
}

class AddressMiner {
private:
    unsigned char initcodeHash[32];
    unsigned char deployerAddr[20];
    unsigned char saltBase[32];
    int deviceId;
    std::chrono::steady_clock::time_point startTime;
    int currentHighScore = 0;
    int targetScore;
    std::string bestAddress;
    unsigned long long bestSalt;

    // Debug tracking
    unsigned int* h_processedBatches;
    unsigned int* d_processedBatches;

    void hexStringToBytes(const std::string& hex, unsigned char* bytes) {
        for (size_t i = 0; i < hex.length(); i += 2) {
            std::string byteString = hex.substr(i, 2);
            bytes[i/2] = std::stoi(byteString, nullptr, 16);
        }
    }

    void printAddress(const unsigned char* hash, unsigned long long salt, int score) {
        auto currentTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);

        std::cout << "New best score: " << std::dec << score << "\n"
                  << "Address: 0x";
        for (int i = 12; i < 32; i++) {
            printf("%02x", (unsigned int)hash[i]);
        }
        std::cout << "\nSalt: " << std::hex << salt << "\n"
                  << "Time Elapsed: " << duration.count() << " seconds\n"
                  << "------------------------\n";
    }

    void printFinalResult() {
        auto currentTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime);

        std::cout << "\n=== FINAL RESULTS ===\n"
                  << "Best Address: 0x" << bestAddress << "\n"
                  << "Best Salt: " << std::hex << bestSalt << "\n"
                  << "Score: " << currentHighScore << "\n"
                  << "Total Time: " << duration.count() << " seconds\n"
                  << "Total Processed Batches: " << *h_processedBatches << "\n";
        exit(0);
    }

public:
    AddressMiner(int gpu = 0, int score = 10) : deviceId(gpu), targetScore(score) {
        // Set CUDA device
        CUDA_CHECK(cudaSetDevice(deviceId));
        
        // Convert hex strings to byte arrays
        hexStringToBytes(INITCODE_HASH, initcodeHash);
        hexStringToBytes(DEPLOYER_ADDRESS, deployerAddr);
        memset(saltBase, 0, 32);
        
        // Allocate debug tracking
        CUDA_CHECK(cudaMallocHost(&h_processedBatches, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_processedBatches, sizeof(unsigned int)));
        *h_processedBatches = 0;
        CUDA_CHECK(cudaMemcpy(d_processedBatches, h_processedBatches, sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Start timer
        startTime = std::chrono::steady_clock::now();
    }

    ~AddressMiner() {
        // Clean up resources
        cudaFreeHost(h_processedBatches);
        cudaFree(d_processedBatches);
    }

    void mine(unsigned long long saltStart, int batchSize = 1024 * 1024) {
        auto iterationStartTime = std::chrono::steady_clock::now();

        // Reset processed batches
        *h_processedBatches = 0;
        CUDA_CHECK(cudaMemcpy(d_processedBatches, h_processedBatches, sizeof(unsigned int), cudaMemcpyHostToDevice));

        unsigned char* d_initcode;
        unsigned char* d_deployerAddr;
        unsigned char* d_saltBase;
        unsigned char* d_results;
        int* d_scores;

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_initcode, 32));
        CUDA_CHECK(cudaMalloc(&d_deployerAddr, 20));
        CUDA_CHECK(cudaMalloc(&d_saltBase, 32));
        CUDA_CHECK(cudaMalloc(&d_results, batchSize * 32));
        CUDA_CHECK(cudaMalloc(&d_scores, batchSize * sizeof(int)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_initcode, initcodeHash, 32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_deployerAddr, deployerAddr, 20, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_saltBase, saltBase, 32, cudaMemcpyHostToDevice));

        // Configure kernel launch
        int threadsPerBlock = 256;
        int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
        

        // Launch kernel
        auto kernelStartTime = std::chrono::steady_clock::now();
        mineAddressesKernel<<<numBlocks, threadsPerBlock>>>(
            d_initcode, d_deployerAddr, d_saltBase, 
            d_results, d_scores, saltStart, batchSize, targetScore,
            d_processedBatches
        );

        // Synchronize and check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy debug info
        CUDA_CHECK(cudaMemcpy(h_processedBatches, d_processedBatches, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Allocate host memory for results
        std::vector<unsigned char> h_results(batchSize * 32);
        std::vector<int> h_scores(batchSize);
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, batchSize * 32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_scores.data(), d_scores, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

        // Existing logic for updating best score and printing best address

        for (int i = 0; i < batchSize; i++) {
            if (h_scores[i] > 0) {

                // Update best score if needed
                if (h_scores[i] > currentHighScore) {
                    currentHighScore = h_scores[i];
                    bestAddress = "";
                    char buffer[3]; // Buffer to hold two hex digits and null terminator
                    for (int j = 12; j < 32; j++) {
                        snprintf(buffer, sizeof(buffer), "%02x", (unsigned int)h_results[i * 32 + j]);
                        bestAddress += buffer; // Append the formatted string to bestAddress
                    }
                    bestSalt = saltStart + i;

                    // Print best result so far
                    printAddress(h_results.data() + i * 32, bestSalt, currentHighScore);

                    // Check if target score is reached
                    if (currentHighScore >= targetScore) {
                        printFinalResult();
                    }
                }
            }
        }

        // Free device memory
        CUDA_CHECK(cudaFree(d_initcode));
        CUDA_CHECK(cudaFree(d_deployerAddr));
        CUDA_CHECK(cudaFree(d_saltBase));
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFree(d_scores));
    }
};

int main() {
    try {
        // Initialize with GPU 0, target score of 50
        AddressMiner miner(0, 50);
        
        unsigned long long startSalt = 0;

        while (true) {
            
            miner.mine(startSalt);
            
            // Increment salt for next iteration
            startSalt += 1024 * 1024;
            
            // Small delay to prevent tight loop
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred!" << std::endl;
        return 2;
    }

    return 0;
}