#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#include <fstream>

namespace cg = cooperative_groups;

#define CHK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t r; asm("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}

__device__ __forceinline__ uint32_t get_clusterid() {
    uint32_t r; asm("mov.u32 %0, %%clusterid.x;" : "=r"(r)); return r;
}

extern __shared__ char smem[];

__global__ void gpc_query_kernel(uint32_t *out) {
    if (threadIdx.x == 0) smem[0] = 1;
    
    cg::cluster_group cluster = cg::this_cluster();
    
    if (threadIdx.x == 0) {
        out[blockIdx.x * 2 + 0] = get_smid();
        out[blockIdx.x * 2 + 1] = get_clusterid(); //blockIdx.x / cluster.num_blocks();
    }
    
    cluster.sync();
}

struct UF {
    std::map<uint32_t, uint32_t> p;
    uint32_t find(uint32_t x) {
        if (p.find(x) == p.end()) p[x] = x;
        return p[x] == x ? x : p[x] = find(p[x]);
    }
    void unite(uint32_t a, uint32_t b) { p[find(a)] = find(b); }
};

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.major < 9) {
        fprintf(stderr, "error: requires sm_90+\n");
        return 1;
    }

    int num_sms = prop.multiProcessorCount;
    int max_smem = prop.sharedMemPerBlockOptin;
    
    CHK(cudaFuncSetAttribute(gpc_query_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    // CHK(cudaFuncSetAttribute(gpc_query_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    int max_cluster = 8;
    {
        cudaLaunchConfig_t cfg = {};
        cfg.blockDim = dim3(32, 1, 1);
        cfg.gridDim = dim3(128, 1, 1);
        cfg.dynamicSmemBytes = max_smem;
        if (cudaOccupancyMaxPotentialClusterSize(&max_cluster, (void*)gpc_query_kernel, &cfg) != cudaSuccess)
            max_cluster = 2;
    }
    if (max_cluster < 2) max_cluster = 8;

    UF uf;
    std::set<uint32_t> all_sms;

    std::vector<std::vector<uint32_t>> sm_associativity_counts(num_sms, std::vector<uint32_t>(num_sms, 0));
    constexpr int nblocks = 256;
    // int csz = 8;

    for (int iter=0; iter < 10000; iter++) {
        for (int csz = 2; csz <= max_cluster; csz=csz*2) {
            size_t out_sz = nblocks * 2 * sizeof(uint32_t);
            
            uint32_t *d_out;
            CHK(cudaMalloc(&d_out, out_sz));
            CHK(cudaMemset(d_out, 0xff, out_sz));
            
            cudaLaunchConfig_t config = {};
            config.gridDim = dim3(nblocks, 1, 1);
            config.blockDim = dim3(32, 1, 1);
            config.dynamicSmemBytes = max_smem;

            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {(unsigned)csz, 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;

            CHK(cudaLaunchKernelEx(&config, gpc_query_kernel, d_out));
            CHK(cudaDeviceSynchronize());

            std::vector<uint32_t> h_out(nblocks * 2);
            CHK(cudaMemcpy(h_out.data(), d_out, out_sz, cudaMemcpyDeviceToHost));
            CHK(cudaFree(d_out));

            std::map<uint32_t, std::vector<uint32_t>> clusters;
            for (int b = 0; b < nblocks; b++) {
                uint32_t smid = h_out[b * 2 + 0];
                uint32_t cid = h_out[b * 2 + 1];
                if (smid != 0xffffffff) {
                    clusters[cid].push_back(smid);
                    all_sms.insert(smid);
                }
            }
            
            for (auto &[cid, sms] : clusters)
                for (size_t i = 1; i < sms.size(); i++)
                    uf.unite(sms[0], sms[i]);

            for (auto &[cid, sms] : clusters) {
                for (size_t i = 0; i < sms.size(); i++) {
                    for (size_t j = 0; j < sms.size(); j++) {
                        // if (i == j) continue;
                        sm_associativity_counts[sms[i]][sms[j]]++;
                    }
                }
            }
        }
    }

    if (all_sms.size() != (size_t)num_sms) {
        fprintf(stderr, "error: detected %zu SMs, expected %d\n", all_sms.size(), num_sms);
        // return 1;
    }

    std::map<uint32_t, std::vector<uint32_t>> gpcs;
    for (uint32_t s : all_sms)
        gpcs[uf.find(s)].push_back(s);

    printf("GPC SM groups:\n");
    int gpc_index = 0;
    for (auto &[_, sms] : gpcs) {
        printf("GPC %d:", gpc_index++);
        for (uint32_t sm : sms)
            printf(" %u", sm);
        printf("\n");
    }

    std::vector<int> tpcs;
    for (auto &[_, sms] : gpcs)
        tpcs.push_back(sms.size() / 2);
    std::sort(tpcs.rbegin(), tpcs.rend());

    printf("[");
    for (size_t i = 0; i < tpcs.size(); i++)
        printf("%s%d", i ? ", " : "", tpcs[i]);
    printf("]\n");

    std::ofstream csv_file("sm_associativity_counts.csv");

    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file\n";
        return 1;
    }

    // Header row
    csv_file << ",";
    for (int col_index = 0; col_index < num_sms; col_index++) {
        csv_file << "SM" << col_index;
        if (col_index != num_sms - 1) {
            csv_file << ",";
        }
    }
    csv_file << "\n";

    // Data rows
    for (int row_index = 0; row_index < num_sms; row_index++) {
        csv_file << "SM" << row_index << ",";

        for (int col_index = 0; col_index < num_sms; col_index++) {
            csv_file << sm_associativity_counts[row_index][col_index];

            if (col_index != num_sms - 1) {
                csv_file << ",";
            }
        }

        csv_file << "\n";
    }

    csv_file.close();

    return 0;
}
