

function costfunction_ode(ψ0,
                         eigvals,
                         signal, 
                         n_sites, 
                         drives, 
                         T, 
                         Cost_ham;
                         tol_ode=1e-8)

    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives,eigvals,tol_ode=tol_ode)
    return real(ψ_ode'*Cost_ham*ψ_ode),  ψ_ode
end

function costfunction_direct_exponentiation(ψ0, 
                                            eigvals,
                                            signal, 
                                            n_sites, 
                                            drives, 
                                            T,  
                                            n_trotter_steps, 
                                            Cost_ham)

    ψ_direct = evolve_direct_exponentiation(ψ0, T, signal, n_sites, drives, eigvals, n_trotter_steps)
    return real(ψ_direct'*Cost_ham*ψ_direct), ψ_direct
end
