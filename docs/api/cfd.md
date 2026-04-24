# `ooc_optimizer.cfd`

::: ooc_optimizer.cfd.scalar
    options:
        members:
            - analytic_ad_1d
            - run_scalar_transport
            - run_scalar_verification_1d
            - extract_concentration_field
            - write_scalar_boundary_file
            - set_transport_diffusivity
            - set_scalar_controldict

::: ooc_optimizer.cfd.solver
    options:
        members:
            - evaluate_cfd

::: ooc_optimizer.cfd.metrics
    options:
        members:
            - extract_v2_metrics
            - extract_metrics
