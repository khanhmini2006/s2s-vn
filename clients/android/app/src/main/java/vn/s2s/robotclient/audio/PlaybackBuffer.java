package vn.s2s.robotclient.audio;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Bộ đệm audio PCM16 nằm giữa luồng WebSocket (ghi vào) và luồng phát loa (đọc ra).
 *
 * <p>Port từ client tham chiếu {@code src/s2s_vn/talk.py:28-61}.
 *
 * <p>Hợp đồng cốt lõi: khi buffer cạn, phần còn lại của mảng đích được ghi ZERO
 * (im lặng) chứ không để rác và không ném exception — loa không bao giờ được đói
 * dữ liệu giữa lúc callback audio đang chạy.
 *
 * <p>Mọi phương thức đều {@code synchronized}: {@code append()} chạy trên luồng
 * WebSocket còn {@code read()} chạy trên callback của AudioTrack, mà {@link ArrayDeque}
 * không thread-safe. Điều này KHÔNG có test bảo vệ — đã thử viết một test hai luồng
 * và xác minh nó không bắt được lỗi (gỡ hết {@code synchronized} vẫn xanh 15/15 lần),
 * nên đã bỏ thay vì giữ một test tạo cảm giác an toàn giả. Đừng gỡ {@code synchronized}
 * chỉ vì thấy test vẫn xanh.
 */
public class PlaybackBuffer {

    /** Các chunk chờ phát, theo thứ tự nhận được. */
    private final Deque<byte[]> chunks = new ArrayDeque<>();

    /** Số byte đã tiêu thụ trong chunk ở đầu hàng đợi. */
    private int offsetTrongChunkDau;

    /** Thêm audio vừa nhận từ server vào cuối hàng đợi. */
    public synchronized void append(byte[] pcm) {
        if (pcm != null && pcm.length > 0) {
            chunks.addLast(pcm);
        }
    }

    /**
     * Xoá sạch audio chưa phát — dùng cho barge-in khi người dùng nói đè lên
     * câu robot đang đọc.
     */
    public synchronized void clear() {
        chunks.clear();
        offsetTrongChunkDau = 0;
    }

    /** Lấy audio ra để phát; thiếu bao nhiêu thì ghi im lặng bấy nhiêu. */
    public synchronized void read(byte[] out) {
        int daGhi = 0;

        while (daGhi < out.length && !chunks.isEmpty()) {
            byte[] dau = chunks.peekFirst();
            int conLaiTrongChunk = dau.length - offsetTrongChunkDau;
            int layDuoc = Math.min(conLaiTrongChunk, out.length - daGhi);

            System.arraycopy(dau, offsetTrongChunkDau, out, daGhi, layDuoc);
            daGhi += layDuoc;
            offsetTrongChunkDau += layDuoc;

            if (offsetTrongChunkDau >= dau.length) {
                chunks.removeFirst();
                offsetTrongChunkDau = 0;
            }
        }

        // Cạn giữa chừng -> phần còn lại là im lặng, ghi đè rác có sẵn.
        java.util.Arrays.fill(out, daGhi, out.length, (byte) 0);
    }
}
