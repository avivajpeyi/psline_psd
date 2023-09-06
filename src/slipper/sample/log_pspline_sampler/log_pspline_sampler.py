from slipper.sample.base_sampler import BaseSampler
import numpy as np

from slipper.sample.base_sampler import BaseSampler, _tune_proposal_distribution
from slipper.splines.initialisation import knot_locator
from slipper.splines.p_splines import PSplines

from .bayesian_functions import lpost, sample_φδτ

class LogPsplineSampler(BaseSampler):
    def _init_mcmc(self) -> None:
        """Initialises the self.samples with the itial values of the MCMC"""

        # init spline model
        sk = self.spline_kwargs
        knots = knot_locator(self.data, self.n_basis, sk["degree"], sk["eqSpaced"])
        self.spline_model = PSplines(
            knots=knots,
            degree=sk["degree"],
            diffMatrixOrder=sk["diffMatrixOrder"],
            all_knots_penalty_matrix=True,
        )

        # init samples
        self.samples = dict(
            w=np.zeros((self.n_steps, self.n_basis)),
            φ=np.zeros(self.n_steps),
            δ=np.zeros(self.n_steps),
            τ=np.zeros(self.n_steps),
            proposal_sigma=np.zeros(self.n_steps),
            acceptance_fraction=np.zeros(self.n_steps),
        )

        sk = self.sampler_kwargs
        self.samples["τ"][0] = np.var(self.data) / (2 * np.pi)
        self.samples["δ"][0] = sk["δα"] / sk["δβ"]
        self.samples["φ"][0] = sk["φα"] / (sk["φβ"] * self.samples["δ"][0])
        self.samples["w"][0, :] = self.spline_model.guess_weights(self.data).ravel()
        self.samples["proposal_sigma"][0] = 1
        self.samples["acceptance_fraction"][0] = 0.4
        self.samples["lpost_trace"] = np.zeros(self.n_steps)

        self.args = [
            self.n_basis,
            self.samples["w"][0],
            self.samples["τ"][0],
            self.sampler_kwargs["τα"],
            self.sampler_kwargs["τβ"],
            self.samples["φ"][0],
            self.sampler_kwargs["φα"],
            self.sampler_kwargs["φβ"],
            self.samples["δ"][0],
            self.sampler_kwargs["δα"],
            self.sampler_kwargs["δβ"],
            self.data,
            self.spline_model,
        ]

    def _mcmc_step(self, itr):
        k = self.n_basis
        aux = np.arange(0, k)
        np.random.shuffle(aux)
        accept_frac = self.samples["acceptance_fraction"][itr - 1]
        sigma = self.samples["proposal_sigma"][itr - 1]

        self.args = [
            self.n_basis,
            self.samples["w"][itr - 1, :],
            self.samples["τ"][itr - 1],
            self.sampler_kwargs["τα"],
            self.sampler_kwargs["τβ"],
            self.samples["φ"][itr - 1],
            self.sampler_kwargs["φα"],
            self.sampler_kwargs["φβ"],
            self.samples["δ"][itr - 1],
            self.sampler_kwargs["δα"],
            self.sampler_kwargs["δβ"],
            self.data,
            self.spline_model,
        ]

        w, τ, φ, δ = self.args[1], self.args[2], self.args[5], self.args[8]
        w_star = self.args[1].copy()
        for _ in range(self.thin):
            lpost_store = lpost(*self.args)
            # 1. explore the parameter space for new V
            w, w_star, accept_frac, sigma = _tune_proposal_distribution(
                aux, accept_frac, sigma, w, w_star, lpost_store, self.args, lpost
            )

            # 2. sample new values for φ, δ, τ
            φ, δ, τ = sample_φδτ(*self.args)
            self.args[1] = w
            self.args[2] = τ
            self.args[5] = φ
            self.args[8] = δ

        # 3. store the new values
        self.samples["φ"][itr] = φ
        self.samples["δ"][itr] = δ
        self.samples["τ"][itr] = τ
        self.samples["w"][itr, :] = w
        self.samples["proposal_sigma"][itr] = sigma
        self.samples["acceptance_fraction"][itr] = accept_frac
        self.samples["lpost_trace"][itr] = lpost_store



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
