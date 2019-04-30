from midi_to_matrix import midiToNoteStateMatrix, noteStateMatrixToMidi
import numpy as np
m = midiToNoteStateMatrix("./midi_files/transposed_01allema.mid")
m = np.array(m[:250])
m = m.reshape((250, 156))
print(m.shape)
m = m.reshape((250, 78, 2))
print(m.tolist())
noteStateMatrixToMidi(m, "allema_test")