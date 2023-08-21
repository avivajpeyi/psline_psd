from slipper.sample.base_sampler import BaseSampler


class LogPsplineSampler(BaseSampler):
    def _init_mcmc(self) -> None:
        pass

    def _mcmc_step(self, itr):
        pass


# Metropolis-within-Gibbs sampler
#
#
# for (j in 1:(N-1)){
# #  print("######################")
# # print(j)
# adj    = (j - 1) * thin;
#
# V.star = V.store;  # proposal value
#
# aux    = sample(k1);  # positions to be changed in the thining loop
#
# # Thining
# for (i in 1:thin) {
#
#     iter = i + adj;
#
# if (iter % % printIter == 0)
# {
#     cat(paste("Iteration", iter, ",", "Time elapsed",
#               round( as.numeric(proc.time()[1] - ptime) / 60, 2),
# "minutes"), "\n")
# }
#
# f.store < - lpost(omega,
#                   FZ,
#                   k,
#                   V.store,  # parameter
#                   tau.store,  # parameter
#                   tau.alpha,
#                   tau.beta,
#                   phi.store,  # parameter
#                   phi.alpha,
#                   phi.beta,
#                   delta.store,  # parameter
#                   delta.alpha,
#                   delta.beta,
#                   P,
#                   pdgrm,
#                   degree,
#                   db.list,
#                   spec_ar)
#
# ##############
# ### WEIGHT ###
# ##############
#
# # aux     = sample(k1);
#
# # tunning proposal distribution
#
# if (count < 0.30)
# {  # increasing acceptance pbb
#
#     sigma = sigma * 0.90;  # decreasing proposal moves
#
# } else if (count > 0.50){  # decreasing acceptance pbb
#
# sigma = sigma * 1.1;  # increasing proposal moves
#
# }
#
# if (count_tau < 0.30){  # increasing acceptance pbb
#
# sigta = sigta * 0.90;  # decreasing proposal moves
#
# } else if (count_tau > 0.50){  # decreasing acceptance pbb
#
# sigta = sigta * 1.1;  # increasing proposal moves
#
# }
#
# count = 0;  # ACCEPTANCE PROBABILITY
# count_tau = 0;
#
# for (g in 1:k1){
#
#     pos = aux[g];
#
# V.star[pos] = V.store[pos] + sigma * Zs[iter, g];
#
# f.V.star < - lpost(omega,
#                    FZ,
#                    k,
#                    V.star,  # proposal value
#                    tau.store,
#                    tau.alpha,
#                    tau.beta,
#                    phi.store,
#                    phi.alpha,
#                    phi.beta,
#                    delta.store,
#                    delta.alpha,
#                    delta.beta,
#                    P,
#                    pdgrm,
#                    degree,
#                    db.list,
#                    spec_ar)
#
# # log posterior for previous iteration
# # f.V <- f.store;
#
# # Accept/reject
#
# alpha1 < - min(0, f.V.star$lp - f.store$lp);  # log acceptance ratio
#
# if (Us[iter, g] < alpha1) {
#
# V.store[pos] < - V.star[pos];  # Accept W.star
# f.store < - f.V.star;
# count < - count + 1;  # acceptance probability
#
# } else {
#
# V.star[pos] = V.store[pos];  # reseting proposal value
#
# }
#
# ###########
# ### tau ###
# ###########
#
# tau.star = tau.store + sigta * Zt[iter, g];
#
# f.tau.star < - lpost(omega,
# FZ,
# k,
# V.store,
# tau.star,  # proposal value
# tau.alpha,
# tau.beta,
# phi.store,
# phi.alpha,
# phi.beta,
# delta.store,
# delta.alpha,
# delta.beta,
# P,
# pdgrm,
# degree,
# db.list,
# spec_ar)
#
# alpha_tau < - min(0, f.tau.star$lp - f.store$lp);  # log acceptance ratio
#
# if (Ut[iter, g] < alpha_tau){
#
# tau.store < - tau.star;  # Accept tau.star
# f.store < - f.tau.star;
# count_tau < - count_tau + 1;  # acceptance probability
#
# }
#
# }  # End updating weights
#
# count       = count / k1;
# Count[iter] = count;  # Acceptance probability
#
# count_tau = count_tau / k1;
# Count_tau = c(Count_tau, count_tau);
#
# ###########
# ### phi ###
# ###########
#
# phi.store = stats::
#     rgamma(1, shape=k / 2 + phi.alpha,
#            rate=phi.beta * delta.store + t(V.store) % * % P % * % V.store / 2);
#
# #############
# ### delta ###
# #############
#
# delta.store = stats::rgamma(1, shape=phi.alpha + delta.alpha,
#                             rate=phi.beta * phi.store + delta.beta);
#
# }  # End thining
#
# ######################
# ### Storing values ###
# ######################
#
# phi[j + 1] = phi.store;
# delta[j + 1] = delta.store;
# tau[j + 1] = tau.store;
# V = cbind(V, V.store);
#
# ### ###
#
# }  # END: MCMC loop
