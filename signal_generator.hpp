#pragma once

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>


using namespace std;


class signal_generator {
public:
    struct minmax_list_item_t {
        int idx;
        bool is_max;
        float value;
    };
    typedef vector<minmax_list_item_t> minmax_list_t;

    typedef vector<int> int_array_t;

    struct sigstat_array_item_t {
        bool sigstat_long;
        bool sigstat_short;
    };
    typedef vector<sigstat_array_item_t> sigstat_array_t;

public:
    vector<float> m_arr;
    int m_arr_size;
    int m_major_ws;
    int m_minor_ws;

public:
    signal_generator(const vector<float> &arr, int major_ws, int minor_ws) 
        : m_arr(arr)
        , m_arr_size(arr.size())
        , m_major_ws(major_ws)
        , m_minor_ws(minor_ws) {
    }

public:
    sigstat_array_t build_state_array(int holdings) {
        auto major_envelop_seq_array = build_envelop_seq_array(make_minmax_list(m_major_ws));
        auto minor_envelop_seq_array = build_envelop_seq_array(make_minmax_list(m_minor_ws));
        auto envelop_count_array = build_envelop_count_array(major_envelop_seq_array, minor_envelop_seq_array);
        sigstat_array_t sigstat;
        sigstat.resize(m_arr_size);

        bool sigstat_long = false;
        int long_remain = 0;
        bool sigstat_short = false;
        int short_remain = 0;
        for (int i = 0; i < m_arr_size; i++) {
            if (major_envelop_seq_array[i] > 0 && minor_envelop_seq_array[i] == 2 && envelop_count_array[i] == 2) {
                sigstat_long = true;
                long_remain = holdings;
            }
            else {
                long_remain -= 1;
                if (long_remain < 0)
                    sigstat_long = false;
            }
            sigstat[i].sigstat_long = sigstat_long;
            if (major_envelop_seq_array[i] < 0 && minor_envelop_seq_array[i] == -2 && envelop_count_array[i] == -2) {
                sigstat_short = true;
                short_remain = holdings;
            }
            else {
                short_remain -= 1;
                if (short_remain < 0)
                    sigstat_short = false;
            }
            sigstat[i].sigstat_short = sigstat_short;
        }

        return sigstat;
    }

protected:
    minmax_list_t make_minmax_list(int winsize) {
        auto min_list = make_envelop_list<less_equal<>>(winsize);
        auto max_list = make_envelop_list<greater_equal<>>(winsize);

        minmax_list_t minmax_list;
        for(auto i: max_list) {
            minmax_list.push_back({i, true, m_arr[i]});
        }
        for(auto i: min_list) {
            minmax_list.push_back({i, false, m_arr[i]});
        }
        sort(minmax_list.begin(), minmax_list.end(), [] (const auto& a, const auto &b) {
            return a.idx < b.idx;
        });
        if (minmax_list.empty()) 
            return minmax_list;

        minmax_list_t result;
        bool current_sign = minmax_list[0].is_max;
        minmax_list_item_t current_row = minmax_list[0];
        for (const auto& row : minmax_list) {
            if (row.is_max != current_sign) {
                result.push_back(current_row);
                current_sign = row.is_max;
                current_row = row;
            }
            if ((current_sign == true && row.value >= current_row.value) ||
                (current_sign == false && row.value <= current_row.value)) {
                current_row = row;
            }
        }
        result.push_back(current_row);
        return result;
    }

    int_array_t build_envelop_seq_array(const minmax_list_t& minmax_list) {
        int_array_t result;
        result.resize(m_arr_size);
        int current_value = minmax_list[0].is_max > 0 ? -1 : 1;
        for (auto it = minmax_list.begin(); it != minmax_list.end(); it++) {
            int start_idx = it->idx;
            auto it_next = it + 1;
            int end_idx = (it_next == minmax_list.end()) ? m_arr_size : it_next->idx;
            for (int i = start_idx; i < end_idx; i++) {
                result[i] = current_value * (i - start_idx + 1);
            }
            current_value *= -1;
        }
        return result;
    }

    int_array_t build_envelop_count_array(const int_array_t& major_seq_array, const int_array_t& minor_seq_array) {
        int_array_t result;
        result.resize(m_arr_size);
        int plus_cnt = 0;
        int minus_cnt = 0;
        for (int i = 0; i < m_arr_size; i++) {
            int major_seq = major_seq_array[i];
            int minor_seq = minor_seq_array[i];
            if (major_seq == 1 or major_seq == -1) {
                minus_cnt = 0;
                plus_cnt = 0;
            }
            if (minor_seq < 0) {
                if (minor_seq == -1)
                    minus_cnt -= 1;
                result[i] = minus_cnt;
            }
            else if (minor_seq > 0) {
                if (minor_seq == 1)
                    plus_cnt += 1;
                result[i] = plus_cnt;
            }
        }
        return result;
    }

private:
    template <typename Compare>
    int find_local_envelop(int winsize, int istart=0) {
        int ubound = m_arr_size - 1;
        int iend = min(ubound, istart + winsize);
        int icur = istart + 1;
        while (icur <= iend) {
            if (Compare()(m_arr[icur], m_arr[istart])) {
                istart = icur;
                iend = min(ubound, istart + winsize);
            }
            icur += 1;
        }
        return istart;
    }

    template <typename Compare>
    vector<int> make_envelop_list(int winsize) {
        vector<int> result;
        int istart = 0;
        int idx = find_local_envelop<Compare>(winsize, istart);
        while (idx < m_arr_size) {
            if (istart != idx)
                result.push_back(idx);
            istart = idx + winsize + 1;
            idx = find_local_envelop<Compare>(winsize, istart);
        }
        return result;
    }
};
