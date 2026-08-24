package vn.s2s.robotclient.audio;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Test cho {@link PlaybackBuffer} — bộ đệm audio phát ra loa.
 *
 * <p>Port hành vi từ lớp {@code PlaybackBuffer} của client tham chiếu
 * {@code src/s2s_vn/talk.py:28-61}.
 */
public class PlaybackBufferTest {

    /** Đọc ra đúng những gì đã ghi vào, giữ nguyên thứ tự. */
    @Test
    public void docRaDungThuTuDaGhi() {
        PlaybackBuffer buffer = new PlaybackBuffer();

        buffer.append(new byte[] {1, 2, 3});
        buffer.append(new byte[] {4, 5, 6});

        byte[] out = new byte[6];
        buffer.read(out);

        assertArrayEquals(new byte[] {1, 2, 3, 4, 5, 6}, out);
    }

    /**
     * Buffer cạn giữa chừng: phần đọc được giữ nguyên, phần còn lại ghi ZERO.
     *
     * <p>Đây là invariant chống underrun quan trọng nhất — loa không bao giờ được
     * đói dữ liệu. Thà phát im lặng còn hơn phát rác hoặc ném exception giữa lúc
     * callback audio đang chạy.
     */
    @Test
    public void canGiuaChungThiGhiSilencePhanConLai() {
        PlaybackBuffer buffer = new PlaybackBuffer();
        buffer.append(new byte[] {7, 8, 9});

        // Xin 6 byte nhưng chỉ có 3.
        byte[] out = new byte[6];
        java.util.Arrays.fill(out, (byte) 0x5A); // rác có sẵn, phải bị ghi đè
        buffer.read(out);

        assertArrayEquals(new byte[] {7, 8, 9, 0, 0, 0}, out);
    }

    /** Buffer rỗng hoàn toàn: toàn bộ mảng đích là im lặng. */
    @Test
    public void bufferRongThiToanImLang() {
        PlaybackBuffer buffer = new PlaybackBuffer();

        byte[] out = new byte[4];
        java.util.Arrays.fill(out, (byte) 0x5A);
        buffer.read(out);

        assertArrayEquals(new byte[] {0, 0, 0, 0}, out);
    }

    /**
     * {@code clear()} xoá sạch audio chưa phát.
     *
     * <p>Dùng khi nhận event {@code input_audio_buffer.speech_started} — người dùng
     * bắt đầu nói đè lên câu robot đang đọc, phần trả lời cũ phải im ngay
     * ({@code talk.py:95}).
     */
    @Test
    public void clearXoaSachAudioChuaPhat() {
        PlaybackBuffer buffer = new PlaybackBuffer();
        buffer.append(new byte[] {1, 2, 3, 4});

        buffer.clear();

        byte[] out = new byte[4];
        buffer.read(out);
        assertArrayEquals(new byte[] {0, 0, 0, 0}, out);
    }

    /**
     * {@code conAudio()} cho biết loa còn phải đọc nữa hay đã hết.
     *
     * <p>Client dùng nó để biết khi nào mở lại mic. Không dùng được sự kiện
     * {@code response.done} của server cho việc này: nó chỉ báo server gửi xong dữ
     * liệu, lúc đó loa vẫn đang phát phần còn tồn trong bộ đệm — mở mic ngay thì thu
     * lại chính tiếng loa.
     */
    @Test
    public void conAudioBaoDungConHayHet() {
        PlaybackBuffer buffer = new PlaybackBuffer();
        assertFalse("buffer mới tạo thì rỗng", buffer.conAudio());

        buffer.append(new byte[] {1, 2, 3, 4});
        assertTrue("vừa nạp thì còn audio", buffer.conAudio());

        // Đọc hết sạch.
        buffer.read(new byte[4]);
        assertFalse("đọc hết rồi thì không còn", buffer.conAudio());

        // Đọc một phần: vẫn còn phần đuôi.
        buffer.append(new byte[] {5, 6, 7, 8});
        buffer.read(new byte[2]);
        assertTrue("mới đọc một nửa thì vẫn còn", buffer.conAudio());

        buffer.clear();
        assertFalse("clear() thì hết", buffer.conAudio());
    }
}
