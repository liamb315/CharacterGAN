#print NumberSequence(generator.predict(np.eye(100)[None,0]).argmx(axis=2).ravel()).decode(text_encoding)


def iterate(iterations, step_size, generator):
    with open(args.log, 'w') as fp:
        for _ in xrange(iterations):
            batch = np.tile(text_encoding.convert_representation([text_encoding.encode('<STR>')]), (args.batch_size, 1))
            y = np.tile([0, 1], (args.sequence_length, args.batch_size, 1))
            loss = rmsprop.train(batch, y, step_size)
            print >> fp,  "Loss[%u]: %f" % (_, loss)
            print "Loss[%u]: %f" % (_, loss)
            fp.flush()
            train_loss.append(loss)

	with open('models/gan-model.pkl', 'wb') as fp:
		pickle.dump(generator.get_state(), fp)
