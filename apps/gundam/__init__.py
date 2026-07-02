"""
GUNDAM/ROOT-backed implementation of the likelihood-sampler interface
described in gunflows.likelihood_sampler.base. This is the only place in
the codebase that imports GUNDAM/ROOT directly; to use a different
likelihood, write a sibling package with the same interface and point the
`sampler_target` config field at it instead.
"""
