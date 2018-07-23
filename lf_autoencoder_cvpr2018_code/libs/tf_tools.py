#
# some TensorFlow tools (mostly found online in threads, thanks to all sources)
#

import tensorflow as tf
#
# this function replaces the stupid standard saver restore,
# it ignores missing variables in the save file.
#
# by RalphMao on GitHub
#
def optimistic_restore(session, save_file):

  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted( [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                      if var.name.split(':')[0] in saved_shapes])
  restore_vars = []
  name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        restore_vars.append(curr_var)

  saver = tf.train.Saver(restore_vars)
  saver.restore(session, save_file)

