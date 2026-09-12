# Pre-execution audit — signed-mass China-5 method harness

- generated: `2026-09-12T10:43:08Z`
- verifier: `verify_preexecution.py` v1.0.0
- verdict: **`SIGNED_MASS_CHINA5_METHOD_HARNESS_READY`**
- checks: 367 run, 0 failed

## What was verified

| # | group | check | result |
|---|---|---|---|
| 1 | readiness | twenty_coordinates | PASS |
| 2 | readiness | coordinate_set_exact | PASS |
| 3 | readiness | no_other_blockers | PASS |
| 4 | readiness | all_ready_or_pending | PASS |
| 5 | readiness | roles_present::GANSU_DA/PatchTST | PASS |
| 6 | readiness | roles_present::GANSU_DA/TimeMixer | PASS |
| 7 | readiness | roles_present::GANSU_DA/iTransformer | PASS |
| 8 | readiness | roles_present::GANSU_DA/LSTM | PASS |
| 9 | readiness | roles_present::SHANDONG_DA/PatchTST | PASS |
| 10 | readiness | roles_present::SHANDONG_DA/TimeMixer | PASS |
| 11 | readiness | roles_present::SHANDONG_DA/iTransformer | PASS |
| 12 | readiness | roles_present::SHANDONG_DA/LSTM | PASS |
| 13 | readiness | roles_present::SHAANXI_DA/PatchTST | PASS |
| 14 | readiness | roles_present::SHAANXI_DA/TimeMixer | PASS |
| 15 | readiness | roles_present::SHAANXI_DA/iTransformer | PASS |
| 16 | readiness | roles_present::SHAANXI_DA/LSTM | PASS |
| 17 | readiness | roles_present::NINGXIA_DA/PatchTST | PASS |
| 18 | readiness | roles_present::NINGXIA_DA/TimeMixer | PASS |
| 19 | readiness | roles_present::NINGXIA_DA/iTransformer | PASS |
| 20 | readiness | roles_present::NINGXIA_DA/LSTM | PASS |
| 21 | readiness | roles_present::QINGHAI_DA/PatchTST | PASS |
| 22 | readiness | roles_present::QINGHAI_DA/TimeMixer | PASS |
| 23 | readiness | roles_present::QINGHAI_DA/iTransformer | PASS |
| 24 | readiness | roles_present::QINGHAI_DA/LSTM | PASS |
| 25 | contracts | features_match_contract::GANSU_DA/PatchTST | PASS |
| 26 | contracts | no_forbidden_feature::GANSU_DA/PatchTST | PASS |
| 27 | contracts | target_not_an_input::GANSU_DA/PatchTST | PASS |
| 28 | contracts | horizon::GANSU_DA/PatchTST | PASS |
| 29 | contracts | role_counts_complete::GANSU_DA/PatchTST | PASS |
| 30 | contracts | sealed_declared_not_open::GANSU_DA/PatchTST | PASS |
| 31 | contracts | source_bytes_intact::GANSU_DA/PatchTST | PASS |
| 32 | cells | materialise::GANSU_DA/PatchTST | PASS |
| 33 | cells | episode_count_agrees::GANSU_DA/PatchTST | PASS |
| 34 | chronology | roles_contiguous_and_ordered::GANSU_DA/PatchTST | PASS |
| 35 | chronology | rows_strictly_increasing::GANSU_DA/PatchTST | PASS |
| 36 | chronology | fitting_pool_is_post_train_only::GANSU_DA/PatchTST | PASS |
| 37 | chronology | history_strictly_causal::GANSU_DA/PatchTST | PASS |
| 38 | chronology | fitting_history_complete::GANSU_DA/PatchTST | PASS |
| 39 | sealing | sealed_absent_and_accounted::GANSU_DA/PatchTST | PASS |
| 40 | legality | cell_audit_all_zero::GANSU_DA/PatchTST | PASS |
| 41 | contracts | features_match_contract::GANSU_DA/TimeMixer | PASS |
| 42 | contracts | no_forbidden_feature::GANSU_DA/TimeMixer | PASS |
| 43 | contracts | target_not_an_input::GANSU_DA/TimeMixer | PASS |
| 44 | contracts | horizon::GANSU_DA/TimeMixer | PASS |
| 45 | contracts | role_counts_complete::GANSU_DA/TimeMixer | PASS |
| 46 | contracts | sealed_declared_not_open::GANSU_DA/TimeMixer | PASS |
| 47 | contracts | source_bytes_intact::GANSU_DA/TimeMixer | PASS |
| 48 | cells | materialise::GANSU_DA/TimeMixer | PASS |
| 49 | cells | episode_count_agrees::GANSU_DA/TimeMixer | PASS |
| 50 | chronology | roles_contiguous_and_ordered::GANSU_DA/TimeMixer | PASS |
| 51 | chronology | rows_strictly_increasing::GANSU_DA/TimeMixer | PASS |
| 52 | chronology | fitting_pool_is_post_train_only::GANSU_DA/TimeMixer | PASS |
| 53 | chronology | history_strictly_causal::GANSU_DA/TimeMixer | PASS |
| 54 | chronology | fitting_history_complete::GANSU_DA/TimeMixer | PASS |
| 55 | sealing | sealed_absent_and_accounted::GANSU_DA/TimeMixer | PASS |
| 56 | legality | cell_audit_all_zero::GANSU_DA/TimeMixer | PASS |
| 57 | contracts | features_match_contract::GANSU_DA/iTransformer | PASS |
| 58 | contracts | no_forbidden_feature::GANSU_DA/iTransformer | PASS |
| 59 | contracts | target_not_an_input::GANSU_DA/iTransformer | PASS |
| 60 | contracts | horizon::GANSU_DA/iTransformer | PASS |
| 61 | contracts | role_counts_complete::GANSU_DA/iTransformer | PASS |
| 62 | contracts | sealed_declared_not_open::GANSU_DA/iTransformer | PASS |
| 63 | contracts | source_bytes_intact::GANSU_DA/iTransformer | PASS |
| 64 | cells | materialise::GANSU_DA/iTransformer | PASS |
| 65 | cells | episode_count_agrees::GANSU_DA/iTransformer | PASS |
| 66 | chronology | roles_contiguous_and_ordered::GANSU_DA/iTransformer | PASS |
| 67 | chronology | rows_strictly_increasing::GANSU_DA/iTransformer | PASS |
| 68 | chronology | fitting_pool_is_post_train_only::GANSU_DA/iTransformer | PASS |
| 69 | chronology | history_strictly_causal::GANSU_DA/iTransformer | PASS |
| 70 | chronology | fitting_history_complete::GANSU_DA/iTransformer | PASS |
| 71 | sealing | sealed_absent_and_accounted::GANSU_DA/iTransformer | PASS |
| 72 | legality | cell_audit_all_zero::GANSU_DA/iTransformer | PASS |
| 73 | contracts | features_match_contract::GANSU_DA/LSTM | PASS |
| 74 | contracts | no_forbidden_feature::GANSU_DA/LSTM | PASS |
| 75 | contracts | target_not_an_input::GANSU_DA/LSTM | PASS |
| 76 | contracts | horizon::GANSU_DA/LSTM | PASS |
| 77 | contracts | role_counts_complete::GANSU_DA/LSTM | PASS |
| 78 | contracts | sealed_declared_not_open::GANSU_DA/LSTM | PASS |
| 79 | contracts | source_bytes_intact::GANSU_DA/LSTM | PASS |
| 80 | cells | materialise::GANSU_DA/LSTM | PASS |
| 81 | cells | episode_count_agrees::GANSU_DA/LSTM | PASS |
| 82 | chronology | roles_contiguous_and_ordered::GANSU_DA/LSTM | PASS |
| 83 | chronology | rows_strictly_increasing::GANSU_DA/LSTM | PASS |
| 84 | chronology | fitting_pool_is_post_train_only::GANSU_DA/LSTM | PASS |
| 85 | chronology | history_strictly_causal::GANSU_DA/LSTM | PASS |
| 86 | chronology | fitting_history_complete::GANSU_DA/LSTM | PASS |
| 87 | sealing | sealed_absent_and_accounted::GANSU_DA/LSTM | PASS |
| 88 | legality | cell_audit_all_zero::GANSU_DA/LSTM | PASS |
| 89 | contracts | features_match_contract::SHANDONG_DA/PatchTST | PASS |
| 90 | contracts | no_forbidden_feature::SHANDONG_DA/PatchTST | PASS |
| 91 | contracts | target_not_an_input::SHANDONG_DA/PatchTST | PASS |
| 92 | contracts | horizon::SHANDONG_DA/PatchTST | PASS |
| 93 | contracts | role_counts_complete::SHANDONG_DA/PatchTST | PASS |
| 94 | contracts | sealed_declared_not_open::SHANDONG_DA/PatchTST | PASS |
| 95 | contracts | source_bytes_intact::SHANDONG_DA/PatchTST | PASS |
| 96 | cells | materialise::SHANDONG_DA/PatchTST | PASS |
| 97 | cells | episode_count_agrees::SHANDONG_DA/PatchTST | PASS |
| 98 | chronology | roles_contiguous_and_ordered::SHANDONG_DA/PatchTST | PASS |
| 99 | chronology | rows_strictly_increasing::SHANDONG_DA/PatchTST | PASS |
| 100 | chronology | fitting_pool_is_post_train_only::SHANDONG_DA/PatchTST | PASS |
| 101 | chronology | history_strictly_causal::SHANDONG_DA/PatchTST | PASS |
| 102 | chronology | fitting_history_complete::SHANDONG_DA/PatchTST | PASS |
| 103 | sealing | sealed_absent_and_accounted::SHANDONG_DA/PatchTST | PASS |
| 104 | legality | cell_audit_all_zero::SHANDONG_DA/PatchTST | PASS |
| 105 | contracts | features_match_contract::SHANDONG_DA/TimeMixer | PASS |
| 106 | contracts | no_forbidden_feature::SHANDONG_DA/TimeMixer | PASS |
| 107 | contracts | target_not_an_input::SHANDONG_DA/TimeMixer | PASS |
| 108 | contracts | horizon::SHANDONG_DA/TimeMixer | PASS |
| 109 | contracts | role_counts_complete::SHANDONG_DA/TimeMixer | PASS |
| 110 | contracts | sealed_declared_not_open::SHANDONG_DA/TimeMixer | PASS |
| 111 | contracts | source_bytes_intact::SHANDONG_DA/TimeMixer | PASS |
| 112 | cells | materialise::SHANDONG_DA/TimeMixer | PASS |
| 113 | cells | episode_count_agrees::SHANDONG_DA/TimeMixer | PASS |
| 114 | chronology | roles_contiguous_and_ordered::SHANDONG_DA/TimeMixer | PASS |
| 115 | chronology | rows_strictly_increasing::SHANDONG_DA/TimeMixer | PASS |
| 116 | chronology | fitting_pool_is_post_train_only::SHANDONG_DA/TimeMixer | PASS |
| 117 | chronology | history_strictly_causal::SHANDONG_DA/TimeMixer | PASS |
| 118 | chronology | fitting_history_complete::SHANDONG_DA/TimeMixer | PASS |
| 119 | sealing | sealed_absent_and_accounted::SHANDONG_DA/TimeMixer | PASS |
| 120 | legality | cell_audit_all_zero::SHANDONG_DA/TimeMixer | PASS |
| 121 | contracts | features_match_contract::SHANDONG_DA/iTransformer | PASS |
| 122 | contracts | no_forbidden_feature::SHANDONG_DA/iTransformer | PASS |
| 123 | contracts | target_not_an_input::SHANDONG_DA/iTransformer | PASS |
| 124 | contracts | horizon::SHANDONG_DA/iTransformer | PASS |
| 125 | contracts | role_counts_complete::SHANDONG_DA/iTransformer | PASS |
| 126 | contracts | sealed_declared_not_open::SHANDONG_DA/iTransformer | PASS |
| 127 | contracts | source_bytes_intact::SHANDONG_DA/iTransformer | PASS |
| 128 | cells | materialise::SHANDONG_DA/iTransformer | PASS |
| 129 | cells | episode_count_agrees::SHANDONG_DA/iTransformer | PASS |
| 130 | chronology | roles_contiguous_and_ordered::SHANDONG_DA/iTransformer | PASS |
| 131 | chronology | rows_strictly_increasing::SHANDONG_DA/iTransformer | PASS |
| 132 | chronology | fitting_pool_is_post_train_only::SHANDONG_DA/iTransformer | PASS |
| 133 | chronology | history_strictly_causal::SHANDONG_DA/iTransformer | PASS |
| 134 | chronology | fitting_history_complete::SHANDONG_DA/iTransformer | PASS |
| 135 | sealing | sealed_absent_and_accounted::SHANDONG_DA/iTransformer | PASS |
| 136 | legality | cell_audit_all_zero::SHANDONG_DA/iTransformer | PASS |
| 137 | contracts | features_match_contract::SHANDONG_DA/LSTM | PASS |
| 138 | contracts | no_forbidden_feature::SHANDONG_DA/LSTM | PASS |
| 139 | contracts | target_not_an_input::SHANDONG_DA/LSTM | PASS |
| 140 | contracts | horizon::SHANDONG_DA/LSTM | PASS |
| 141 | contracts | role_counts_complete::SHANDONG_DA/LSTM | PASS |
| 142 | contracts | sealed_declared_not_open::SHANDONG_DA/LSTM | PASS |
| 143 | contracts | source_bytes_intact::SHANDONG_DA/LSTM | PASS |
| 144 | cells | materialise::SHANDONG_DA/LSTM | PASS |
| 145 | cells | episode_count_agrees::SHANDONG_DA/LSTM | PASS |
| 146 | chronology | roles_contiguous_and_ordered::SHANDONG_DA/LSTM | PASS |
| 147 | chronology | rows_strictly_increasing::SHANDONG_DA/LSTM | PASS |
| 148 | chronology | fitting_pool_is_post_train_only::SHANDONG_DA/LSTM | PASS |
| 149 | chronology | history_strictly_causal::SHANDONG_DA/LSTM | PASS |
| 150 | chronology | fitting_history_complete::SHANDONG_DA/LSTM | PASS |
| 151 | sealing | sealed_absent_and_accounted::SHANDONG_DA/LSTM | PASS |
| 152 | legality | cell_audit_all_zero::SHANDONG_DA/LSTM | PASS |
| 153 | contracts | features_match_contract::SHAANXI_DA/PatchTST | PASS |
| 154 | contracts | no_forbidden_feature::SHAANXI_DA/PatchTST | PASS |
| 155 | contracts | target_not_an_input::SHAANXI_DA/PatchTST | PASS |
| 156 | contracts | horizon::SHAANXI_DA/PatchTST | PASS |
| 157 | contracts | role_counts_complete::SHAANXI_DA/PatchTST | PASS |
| 158 | contracts | sealed_declared_not_open::SHAANXI_DA/PatchTST | PASS |
| 159 | contracts | source_bytes_intact::SHAANXI_DA/PatchTST | PASS |
| 160 | cells | materialise::SHAANXI_DA/PatchTST | PASS |
| 161 | cells | episode_count_agrees::SHAANXI_DA/PatchTST | PASS |
| 162 | chronology | roles_contiguous_and_ordered::SHAANXI_DA/PatchTST | PASS |
| 163 | chronology | rows_strictly_increasing::SHAANXI_DA/PatchTST | PASS |
| 164 | chronology | fitting_pool_is_post_train_only::SHAANXI_DA/PatchTST | PASS |
| 165 | chronology | history_strictly_causal::SHAANXI_DA/PatchTST | PASS |
| 166 | chronology | fitting_history_complete::SHAANXI_DA/PatchTST | PASS |
| 167 | sealing | sealed_absent_and_accounted::SHAANXI_DA/PatchTST | PASS |
| 168 | legality | cell_audit_all_zero::SHAANXI_DA/PatchTST | PASS |
| 169 | contracts | features_match_contract::SHAANXI_DA/TimeMixer | PASS |
| 170 | contracts | no_forbidden_feature::SHAANXI_DA/TimeMixer | PASS |
| 171 | contracts | target_not_an_input::SHAANXI_DA/TimeMixer | PASS |
| 172 | contracts | horizon::SHAANXI_DA/TimeMixer | PASS |
| 173 | contracts | role_counts_complete::SHAANXI_DA/TimeMixer | PASS |
| 174 | contracts | sealed_declared_not_open::SHAANXI_DA/TimeMixer | PASS |
| 175 | contracts | source_bytes_intact::SHAANXI_DA/TimeMixer | PASS |
| 176 | cells | materialise::SHAANXI_DA/TimeMixer | PASS |
| 177 | cells | episode_count_agrees::SHAANXI_DA/TimeMixer | PASS |
| 178 | chronology | roles_contiguous_and_ordered::SHAANXI_DA/TimeMixer | PASS |
| 179 | chronology | rows_strictly_increasing::SHAANXI_DA/TimeMixer | PASS |
| 180 | chronology | fitting_pool_is_post_train_only::SHAANXI_DA/TimeMixer | PASS |
| 181 | chronology | history_strictly_causal::SHAANXI_DA/TimeMixer | PASS |
| 182 | chronology | fitting_history_complete::SHAANXI_DA/TimeMixer | PASS |
| 183 | sealing | sealed_absent_and_accounted::SHAANXI_DA/TimeMixer | PASS |
| 184 | legality | cell_audit_all_zero::SHAANXI_DA/TimeMixer | PASS |
| 185 | contracts | features_match_contract::SHAANXI_DA/iTransformer | PASS |
| 186 | contracts | no_forbidden_feature::SHAANXI_DA/iTransformer | PASS |
| 187 | contracts | target_not_an_input::SHAANXI_DA/iTransformer | PASS |
| 188 | contracts | horizon::SHAANXI_DA/iTransformer | PASS |
| 189 | contracts | role_counts_complete::SHAANXI_DA/iTransformer | PASS |
| 190 | contracts | sealed_declared_not_open::SHAANXI_DA/iTransformer | PASS |
| 191 | contracts | source_bytes_intact::SHAANXI_DA/iTransformer | PASS |
| 192 | cells | materialise::SHAANXI_DA/iTransformer | PASS |
| 193 | cells | episode_count_agrees::SHAANXI_DA/iTransformer | PASS |
| 194 | chronology | roles_contiguous_and_ordered::SHAANXI_DA/iTransformer | PASS |
| 195 | chronology | rows_strictly_increasing::SHAANXI_DA/iTransformer | PASS |
| 196 | chronology | fitting_pool_is_post_train_only::SHAANXI_DA/iTransformer | PASS |
| 197 | chronology | history_strictly_causal::SHAANXI_DA/iTransformer | PASS |
| 198 | chronology | fitting_history_complete::SHAANXI_DA/iTransformer | PASS |
| 199 | sealing | sealed_absent_and_accounted::SHAANXI_DA/iTransformer | PASS |
| 200 | legality | cell_audit_all_zero::SHAANXI_DA/iTransformer | PASS |
| 201 | contracts | features_match_contract::SHAANXI_DA/LSTM | PASS |
| 202 | contracts | no_forbidden_feature::SHAANXI_DA/LSTM | PASS |
| 203 | contracts | target_not_an_input::SHAANXI_DA/LSTM | PASS |
| 204 | contracts | horizon::SHAANXI_DA/LSTM | PASS |
| 205 | contracts | role_counts_complete::SHAANXI_DA/LSTM | PASS |
| 206 | contracts | sealed_declared_not_open::SHAANXI_DA/LSTM | PASS |
| 207 | contracts | source_bytes_intact::SHAANXI_DA/LSTM | PASS |
| 208 | cells | materialise::SHAANXI_DA/LSTM | PASS |
| 209 | cells | episode_count_agrees::SHAANXI_DA/LSTM | PASS |
| 210 | chronology | roles_contiguous_and_ordered::SHAANXI_DA/LSTM | PASS |
| 211 | chronology | rows_strictly_increasing::SHAANXI_DA/LSTM | PASS |
| 212 | chronology | fitting_pool_is_post_train_only::SHAANXI_DA/LSTM | PASS |
| 213 | chronology | history_strictly_causal::SHAANXI_DA/LSTM | PASS |
| 214 | chronology | fitting_history_complete::SHAANXI_DA/LSTM | PASS |
| 215 | sealing | sealed_absent_and_accounted::SHAANXI_DA/LSTM | PASS |
| 216 | legality | cell_audit_all_zero::SHAANXI_DA/LSTM | PASS |
| 217 | contracts | features_match_contract::NINGXIA_DA/PatchTST | PASS |
| 218 | contracts | no_forbidden_feature::NINGXIA_DA/PatchTST | PASS |
| 219 | contracts | target_not_an_input::NINGXIA_DA/PatchTST | PASS |
| 220 | contracts | horizon::NINGXIA_DA/PatchTST | PASS |
| 221 | contracts | role_counts_complete::NINGXIA_DA/PatchTST | PASS |
| 222 | contracts | sealed_declared_not_open::NINGXIA_DA/PatchTST | PASS |
| 223 | contracts | source_bytes_intact::NINGXIA_DA/PatchTST | PASS |
| 224 | cells | materialise::NINGXIA_DA/PatchTST | PASS |
| 225 | cells | episode_count_agrees::NINGXIA_DA/PatchTST | PASS |
| 226 | chronology | roles_contiguous_and_ordered::NINGXIA_DA/PatchTST | PASS |
| 227 | chronology | rows_strictly_increasing::NINGXIA_DA/PatchTST | PASS |
| 228 | chronology | fitting_pool_is_post_train_only::NINGXIA_DA/PatchTST | PASS |
| 229 | chronology | history_strictly_causal::NINGXIA_DA/PatchTST | PASS |
| 230 | chronology | fitting_history_complete::NINGXIA_DA/PatchTST | PASS |
| 231 | sealing | sealed_absent_and_accounted::NINGXIA_DA/PatchTST | PASS |
| 232 | legality | cell_audit_all_zero::NINGXIA_DA/PatchTST | PASS |
| 233 | contracts | features_match_contract::NINGXIA_DA/TimeMixer | PASS |
| 234 | contracts | no_forbidden_feature::NINGXIA_DA/TimeMixer | PASS |
| 235 | contracts | target_not_an_input::NINGXIA_DA/TimeMixer | PASS |
| 236 | contracts | horizon::NINGXIA_DA/TimeMixer | PASS |
| 237 | contracts | role_counts_complete::NINGXIA_DA/TimeMixer | PASS |
| 238 | contracts | sealed_declared_not_open::NINGXIA_DA/TimeMixer | PASS |
| 239 | contracts | source_bytes_intact::NINGXIA_DA/TimeMixer | PASS |
| 240 | cells | materialise::NINGXIA_DA/TimeMixer | PASS |
| 241 | cells | episode_count_agrees::NINGXIA_DA/TimeMixer | PASS |
| 242 | chronology | roles_contiguous_and_ordered::NINGXIA_DA/TimeMixer | PASS |
| 243 | chronology | rows_strictly_increasing::NINGXIA_DA/TimeMixer | PASS |
| 244 | chronology | fitting_pool_is_post_train_only::NINGXIA_DA/TimeMixer | PASS |
| 245 | chronology | history_strictly_causal::NINGXIA_DA/TimeMixer | PASS |
| 246 | chronology | fitting_history_complete::NINGXIA_DA/TimeMixer | PASS |
| 247 | sealing | sealed_absent_and_accounted::NINGXIA_DA/TimeMixer | PASS |
| 248 | legality | cell_audit_all_zero::NINGXIA_DA/TimeMixer | PASS |
| 249 | contracts | features_match_contract::NINGXIA_DA/iTransformer | PASS |
| 250 | contracts | no_forbidden_feature::NINGXIA_DA/iTransformer | PASS |
| 251 | contracts | target_not_an_input::NINGXIA_DA/iTransformer | PASS |
| 252 | contracts | horizon::NINGXIA_DA/iTransformer | PASS |
| 253 | contracts | role_counts_complete::NINGXIA_DA/iTransformer | PASS |
| 254 | contracts | sealed_declared_not_open::NINGXIA_DA/iTransformer | PASS |
| 255 | contracts | source_bytes_intact::NINGXIA_DA/iTransformer | PASS |
| 256 | cells | materialise::NINGXIA_DA/iTransformer | PASS |
| 257 | cells | episode_count_agrees::NINGXIA_DA/iTransformer | PASS |
| 258 | chronology | roles_contiguous_and_ordered::NINGXIA_DA/iTransformer | PASS |
| 259 | chronology | rows_strictly_increasing::NINGXIA_DA/iTransformer | PASS |
| 260 | chronology | fitting_pool_is_post_train_only::NINGXIA_DA/iTransformer | PASS |
| 261 | chronology | history_strictly_causal::NINGXIA_DA/iTransformer | PASS |
| 262 | chronology | fitting_history_complete::NINGXIA_DA/iTransformer | PASS |
| 263 | sealing | sealed_absent_and_accounted::NINGXIA_DA/iTransformer | PASS |
| 264 | legality | cell_audit_all_zero::NINGXIA_DA/iTransformer | PASS |
| 265 | contracts | features_match_contract::NINGXIA_DA/LSTM | PASS |
| 266 | contracts | no_forbidden_feature::NINGXIA_DA/LSTM | PASS |
| 267 | contracts | target_not_an_input::NINGXIA_DA/LSTM | PASS |
| 268 | contracts | horizon::NINGXIA_DA/LSTM | PASS |
| 269 | contracts | role_counts_complete::NINGXIA_DA/LSTM | PASS |
| 270 | contracts | sealed_declared_not_open::NINGXIA_DA/LSTM | PASS |
| 271 | contracts | source_bytes_intact::NINGXIA_DA/LSTM | PASS |
| 272 | cells | materialise::NINGXIA_DA/LSTM | PASS |
| 273 | cells | episode_count_agrees::NINGXIA_DA/LSTM | PASS |
| 274 | chronology | roles_contiguous_and_ordered::NINGXIA_DA/LSTM | PASS |
| 275 | chronology | rows_strictly_increasing::NINGXIA_DA/LSTM | PASS |
| 276 | chronology | fitting_pool_is_post_train_only::NINGXIA_DA/LSTM | PASS |
| 277 | chronology | history_strictly_causal::NINGXIA_DA/LSTM | PASS |
| 278 | chronology | fitting_history_complete::NINGXIA_DA/LSTM | PASS |
| 279 | sealing | sealed_absent_and_accounted::NINGXIA_DA/LSTM | PASS |
| 280 | legality | cell_audit_all_zero::NINGXIA_DA/LSTM | PASS |
| 281 | contracts | features_match_contract::QINGHAI_DA/PatchTST | PASS |
| 282 | contracts | no_forbidden_feature::QINGHAI_DA/PatchTST | PASS |
| 283 | contracts | target_not_an_input::QINGHAI_DA/PatchTST | PASS |
| 284 | contracts | horizon::QINGHAI_DA/PatchTST | PASS |
| 285 | contracts | role_counts_complete::QINGHAI_DA/PatchTST | PASS |
| 286 | contracts | sealed_declared_not_open::QINGHAI_DA/PatchTST | PASS |
| 287 | contracts | source_bytes_intact::QINGHAI_DA/PatchTST | PASS |
| 288 | cells | materialise::QINGHAI_DA/PatchTST | PASS |
| 289 | cells | episode_count_agrees::QINGHAI_DA/PatchTST | PASS |
| 290 | chronology | roles_contiguous_and_ordered::QINGHAI_DA/PatchTST | PASS |
| 291 | chronology | rows_strictly_increasing::QINGHAI_DA/PatchTST | PASS |
| 292 | chronology | fitting_pool_is_post_train_only::QINGHAI_DA/PatchTST | PASS |
| 293 | chronology | history_strictly_causal::QINGHAI_DA/PatchTST | PASS |
| 294 | chronology | fitting_history_complete::QINGHAI_DA/PatchTST | PASS |
| 295 | sealing | sealed_absent_and_accounted::QINGHAI_DA/PatchTST | PASS |
| 296 | legality | cell_audit_all_zero::QINGHAI_DA/PatchTST | PASS |
| 297 | contracts | features_match_contract::QINGHAI_DA/TimeMixer | PASS |
| 298 | contracts | no_forbidden_feature::QINGHAI_DA/TimeMixer | PASS |
| 299 | contracts | target_not_an_input::QINGHAI_DA/TimeMixer | PASS |
| 300 | contracts | horizon::QINGHAI_DA/TimeMixer | PASS |
| 301 | contracts | role_counts_complete::QINGHAI_DA/TimeMixer | PASS |
| 302 | contracts | sealed_declared_not_open::QINGHAI_DA/TimeMixer | PASS |
| 303 | contracts | source_bytes_intact::QINGHAI_DA/TimeMixer | PASS |
| 304 | cells | materialise::QINGHAI_DA/TimeMixer | PASS |
| 305 | cells | episode_count_agrees::QINGHAI_DA/TimeMixer | PASS |
| 306 | chronology | roles_contiguous_and_ordered::QINGHAI_DA/TimeMixer | PASS |
| 307 | chronology | rows_strictly_increasing::QINGHAI_DA/TimeMixer | PASS |
| 308 | chronology | fitting_pool_is_post_train_only::QINGHAI_DA/TimeMixer | PASS |
| 309 | chronology | history_strictly_causal::QINGHAI_DA/TimeMixer | PASS |
| 310 | chronology | fitting_history_complete::QINGHAI_DA/TimeMixer | PASS |
| 311 | sealing | sealed_absent_and_accounted::QINGHAI_DA/TimeMixer | PASS |
| 312 | legality | cell_audit_all_zero::QINGHAI_DA/TimeMixer | PASS |
| 313 | contracts | features_match_contract::QINGHAI_DA/iTransformer | PASS |
| 314 | contracts | no_forbidden_feature::QINGHAI_DA/iTransformer | PASS |
| 315 | contracts | target_not_an_input::QINGHAI_DA/iTransformer | PASS |
| 316 | contracts | horizon::QINGHAI_DA/iTransformer | PASS |
| 317 | contracts | role_counts_complete::QINGHAI_DA/iTransformer | PASS |
| 318 | contracts | sealed_declared_not_open::QINGHAI_DA/iTransformer | PASS |
| 319 | contracts | source_bytes_intact::QINGHAI_DA/iTransformer | PASS |
| 320 | cells | materialise::QINGHAI_DA/iTransformer | PASS |
| 321 | cells | episode_count_agrees::QINGHAI_DA/iTransformer | PASS |
| 322 | chronology | roles_contiguous_and_ordered::QINGHAI_DA/iTransformer | PASS |
| 323 | chronology | rows_strictly_increasing::QINGHAI_DA/iTransformer | PASS |
| 324 | chronology | fitting_pool_is_post_train_only::QINGHAI_DA/iTransformer | PASS |
| 325 | chronology | history_strictly_causal::QINGHAI_DA/iTransformer | PASS |
| 326 | chronology | fitting_history_complete::QINGHAI_DA/iTransformer | PASS |
| 327 | sealing | sealed_absent_and_accounted::QINGHAI_DA/iTransformer | PASS |
| 328 | legality | cell_audit_all_zero::QINGHAI_DA/iTransformer | PASS |
| 329 | contracts | features_match_contract::QINGHAI_DA/LSTM | PASS |
| 330 | contracts | no_forbidden_feature::QINGHAI_DA/LSTM | PASS |
| 331 | contracts | target_not_an_input::QINGHAI_DA/LSTM | PASS |
| 332 | contracts | horizon::QINGHAI_DA/LSTM | PASS |
| 333 | contracts | role_counts_complete::QINGHAI_DA/LSTM | PASS |
| 334 | contracts | sealed_declared_not_open::QINGHAI_DA/LSTM | PASS |
| 335 | contracts | source_bytes_intact::QINGHAI_DA/LSTM | PASS |
| 336 | cells | materialise::QINGHAI_DA/LSTM | PASS |
| 337 | cells | episode_count_agrees::QINGHAI_DA/LSTM | PASS |
| 338 | chronology | roles_contiguous_and_ordered::QINGHAI_DA/LSTM | PASS |
| 339 | chronology | rows_strictly_increasing::QINGHAI_DA/LSTM | PASS |
| 340 | chronology | fitting_pool_is_post_train_only::QINGHAI_DA/LSTM | PASS |
| 341 | chronology | history_strictly_causal::QINGHAI_DA/LSTM | PASS |
| 342 | chronology | fitting_history_complete::QINGHAI_DA/LSTM | PASS |
| 343 | sealing | sealed_absent_and_accounted::QINGHAI_DA/LSTM | PASS |
| 344 | legality | cell_audit_all_zero::QINGHAI_DA/LSTM | PASS |
| 345 | method | ablations_flip_exactly_one_switch | PASS |
| 346 | method | every_switch_has_exactly_one_ablation | PASS |
| 347 | method | five_registered_configurations | PASS |
| 348 | method | one_configuration_per_distinct_vector | PASS |
| 349 | method | frozen_smallest_is_not_pre_registered | PASS |
| 350 | method | registered_seeds_are_7_17_37 | PASS |
| 351 | method | panel_is_twenty_cells | PASS |
| 352 | method | knn_off_in_the_core_default | PASS |
| 353 | source | no_gate_or_router_class | PASS |
| 354 | source | no_attention | PASS |
| 355 | source | no_nn_module_definition | PASS |
| 356 | source | no_artifact_write | PASS |
| 357 | source | no_optimizer_outside_training | PASS |
| 358 | source | no_host_training | PASS |
| 359 | source | market_literal_branches_exactly_registered | PASS |
| 360 | source | writer_exemptions_are_guarded | PASS |
| 361 | source | no_local_scientific_definition | PASS |
| 362 | source | ban_list_intact | PASS |
| 363 | source | every_core_attr_is_audited | PASS |
| 364 | source | sys_path_only_in_bridge | PASS |
| 365 | core | core_tree_unchanged_during_audit | PASS |
| 366 | core | core_exports_complete | PASS |
| 367 | immutability | read_set_unchanged | PASS |

## Readiness

- coordinates: 20
- `READY_FROZEN`: 20
- `PENDING_EXTERNAL_HOST_ARTIFACT`: 0
- `GANSU_DA/PatchTST` — F=7, episodes=291, source=`LEGACY_PREFLIGHT`, declared sealed days=125, sealed read count=0
- `GANSU_DA/TimeMixer` — F=7, episodes=291, source=`LEGACY_PREFLIGHT`, declared sealed days=125, sealed read count=0
- `GANSU_DA/iTransformer` — F=7, episodes=291, source=`LEGACY_PREFLIGHT`, declared sealed days=125, sealed read count=0
- `GANSU_DA/LSTM` — F=7, episodes=291, source=`LEGACY_PREFLIGHT`, declared sealed days=125, sealed read count=0
- `SHANDONG_DA/PatchTST` — F=9, episodes=1156, source=`CANONICAL_CHINA5`, declared sealed days=496, sealed read count=0
- `SHANDONG_DA/TimeMixer` — F=9, episodes=1156, source=`CANONICAL_CHINA5`, declared sealed days=496, sealed read count=0
- `SHANDONG_DA/iTransformer` — F=9, episodes=1156, source=`CANONICAL_CHINA5`, declared sealed days=496, sealed read count=0
- `SHANDONG_DA/LSTM` — F=9, episodes=1156, source=`CANONICAL_CHINA5`, declared sealed days=496, sealed read count=0
- `SHAANXI_DA/PatchTST` — F=8, episodes=284, source=`CANONICAL_CHINA5`, declared sealed days=122, sealed read count=0
- `SHAANXI_DA/TimeMixer` — F=8, episodes=284, source=`CANONICAL_CHINA5`, declared sealed days=122, sealed read count=0
- `SHAANXI_DA/iTransformer` — F=8, episodes=284, source=`CANONICAL_CHINA5`, declared sealed days=122, sealed read count=0
- `SHAANXI_DA/LSTM` — F=8, episodes=284, source=`CANONICAL_CHINA5`, declared sealed days=122, sealed read count=0
- `NINGXIA_DA/PatchTST` — F=7, episodes=101, source=`CANONICAL_CHINA5`, declared sealed days=43, sealed read count=0
- `NINGXIA_DA/TimeMixer` — F=7, episodes=101, source=`CANONICAL_CHINA5`, declared sealed days=43, sealed read count=0
- `NINGXIA_DA/iTransformer` — F=7, episodes=101, source=`CANONICAL_CHINA5`, declared sealed days=43, sealed read count=0
- `NINGXIA_DA/LSTM` — F=7, episodes=101, source=`CANONICAL_CHINA5`, declared sealed days=43, sealed read count=0
- `QINGHAI_DA/PatchTST` — F=4, episodes=94, source=`CANONICAL_CHINA5`, declared sealed days=41, sealed read count=0
- `QINGHAI_DA/TimeMixer` — F=4, episodes=94, source=`CANONICAL_CHINA5`, declared sealed days=41, sealed read count=0
- `QINGHAI_DA/iTransformer` — F=4, episodes=94, source=`CANONICAL_CHINA5`, declared sealed days=41, sealed read count=0
- `QINGHAI_DA/LSTM` — F=4, episodes=94, source=`CANONICAL_CHINA5`, declared sealed days=41, sealed read count=0

## Legality statement

- target-day price, realised future price and competition-space
  forecast columns are excluded at the `usecols` level, not filtered after
  the fact; the adapter never opens them on the model path.
- `PROTECTED_FINAL` read count: **0**, witnessed independently by
  row accounting, label absence and lookup refusal.
- `DEV_EVAL` rows are materialised but never passed to any fitting call;
  `training.py` receives positional rows and cannot resolve roles.
- thresholds are read from the frozen file and are never recomputed:
  `experiments/evidence/hch_china_host_breadth_expansion_20260912/00_protocol/THRESHOLD_FREEZE.json` (`ADBFFA74F6B8C876…`)

## Immutability of the read set

- files hashed before and after: 79
- added: 0, removed: 0, changed: 0

## What this audit does *not* authorize

- it does not authorize fitting the method on any partition;
- it does not authorize reading `DEV_EVAL` outcomes;
- it does not authorize unsealing `PROTECTED_FINAL`;
- it does not retrain or re-fit any Host, baseline, scaler or threshold.
