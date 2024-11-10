import stim

if __name__ == '__main__':
    circ = stim.Circuit()
    circ.append_operation('H', [0])
    circ.append_operation('CX', [0, 1])

    circ.append_operation('DEPOLARIZE1', [0, 1], 0.01)

    circ.append_operation('M', [0])
    circ.append_operation('M', [1])

    circ.append_operation('DETECTOR', [stim.target_rec(-2), stim.target_rec(-1)], 0)

    circ.append_operation("OBSERVABLE_INCLUDE", stim.target_rec(-2), 0)
    circ.append_operation("OBSERVABLE_INCLUDE", stim.target_rec(-1), 1)
    circ.append_operation("OBSERVABLE_INCLUDE", stim.target_rec(-2), 2)
    circ.append_operation("OBSERVABLE_INCLUDE", stim.target_rec(-1), 2)
    print(circ)

    # model = circ.detector_error_model(decompose_errors=True)
    # matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circ.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)
    # predicted_observables = matching.decode_batch(syndrome)
    # num_errors = np.sum(predicted_observables != actual_observables, axis=0)
    # log_err_rate = num_errors / shots
    # print("logical error_rate", log_err_rate)
    print()
