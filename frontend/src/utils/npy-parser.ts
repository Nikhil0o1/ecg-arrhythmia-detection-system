/* ──────────────────────────────────────────────────────────────
   Minimal .npy parser for 1-D float64 / float32 arrays.
   Works in the browser via FileReader → ArrayBuffer.
   ────────────────────────────────────────────────────────────── */

/**
 * Parse a .npy binary buffer and return a plain number[].
 * Supports little-endian float64 (<f8) and float32 (<f4).
 */
export function parseNpy(buffer: ArrayBuffer): number[] {
  const view = new DataView(buffer);

  // ── Magic: 0x93 N U M P Y ──
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5),
  );
  if (magic !== "\x93NUMPY") {
    throw new Error("Not a valid .npy file (bad magic number).");
  }

  const major = view.getUint8(6);
  // const minor = view.getUint8(7);

  // Header length
  let headerLen: number;
  let dataOffset: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true); // little-endian u16
    dataOffset = 10 + headerLen;
  } else if (major === 2) {
    headerLen = view.getUint32(8, true); // little-endian u32
    dataOffset = 12 + headerLen;
  } else {
    throw new Error(`Unsupported .npy version ${major}.`);
  }

  // Parse header dict (Python literal)
  const decoder = new TextDecoder("ascii");
  const headerStr = decoder.decode(
    new Uint8Array(buffer, major === 1 ? 10 : 12, headerLen),
  );

  // Extract dtype descriptor
  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  if (!descrMatch) throw new Error("Cannot parse dtype from .npy header.");
  const descr = descrMatch[1]!;

  // Determine byte width & typed-array constructor
  let bytesPerElement: number;
  let readFn: (offset: number) => number;

  if (descr.includes("f8")) {
    bytesPerElement = 8;
    readFn = (off: number) => view.getFloat64(off, true);
  } else if (descr.includes("f4")) {
    bytesPerElement = 4;
    readFn = (off: number) => view.getFloat32(off, true);
  } else {
    throw new Error(
      `Unsupported dtype "${descr}". Only float32/float64 are supported.`,
    );
  }

  const dataBytes = buffer.byteLength - dataOffset;
  const count = dataBytes / bytesPerElement;

  const result: number[] = new Array(count);
  for (let i = 0; i < count; i++) {
    result[i] = readFn(dataOffset + i * bytesPerElement);
  }

  return result;
}

/**
 * Read a File as an ArrayBuffer, then parse it as .npy.
 */
export async function readNpyFile(file: File): Promise<number[]> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        resolve(parseNpy(reader.result as ArrayBuffer));
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsArrayBuffer(file);
  });
}
